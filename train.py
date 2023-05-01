import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
import transformers
from datasets import Dataset
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms import UniformTemporalSubsample
from torchvision.transforms import Compose
from transformers.deepspeed import is_deepspeed_zero3_enabled

from video_blip2 import VideoBlip2ForConditionalGeneration

PROMPT = "Question: What is the camera wearer doing? Answer:"
INSTR_PROMPT = "What is the camera wearer doing?"

C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)


def load_ego4d_fho_clip_dataset(annotation_path: str) -> Dataset:
    with open(annotation_path) as f:
        annotations = json.load(f)

    # create a dataset out of all the narrated actions
    return Dataset.from_list(
        [
            {"clip_uid": interval["clip_uid"], **action}
            for video in annotations["videos"]
            for interval in video["annotated_intervals"]
            for action in interval["narrated_actions"]
        ]
    )


def clean_narration_text(narration_text: str) -> dict[str, str]:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    cleaned = re.sub(C_REGEX, "The camera wearer", cleaned)

    return {"cleaned_narration_text": cleaned}


def add_prompt_column(item: dict, instruct_tuned: bool = True) -> dict:
    item["prompt"] = INSTR_PROMPT if instruct_tuned else PROMPT
    return item


def generate_inputs(item: dict, use_decoder_only_lm: bool = True) -> dict:
    """Generate text inputs.

    If using a decoder only language model, we concatenate the prompt
    with "narration_text" into a single input. For seq2seq language
    models, we put the prompt as the input and narration_text as the
    label.
    """
    if use_decoder_only_lm:
        item["input"] = item["prompt"] + " " + item["cleaned_narration_text"]
        return item
    item["input"] = item["prompt"]
    item["labels"] = item["cleaned_narration_text"]
    return item


def batch_tokenize(
    tokenizer: transformers.PreTrainedTokenizer,
    batch_items: dict,
    use_decoder_only_lm: bool = True,
) -> dict:
    """Tokenize text inputs.

    Note that we don't need to do any right shifting here since that's
    handled by the model. For example, T5ForconditionalGeneration right
    shifts the label using the pad token, while GPT2LMHeadModel removes
    the last token, i.e., eos, from the input and removes the first
    token from the label, which is set to be the same as the input by
    DataCollatorForLanguageModeling.
    """
    # NOTE: since examples will be batched by the collator, we don't need to
    # generate attention masks here.
    tokenized = {
        "input_ids": tokenizer(
            batch_items["input"], return_attention_mask=False
        ).input_ids
    }
    if use_decoder_only_lm:
        # append eos
        for input_ids in tokenized["input_ids"]:
            input_ids.append(tokenizer.eos_token_id)
    if "labels" in batch_items:
        tokenized["labels"] = tokenizer(
            batch_items["labels"], return_attention_mask=False
        ).input_ids
    return tokenized


def extract_frames(
    video_path_handler: VideoPathHandler,
    image_processor: transformers.BlipImageProcessor,
    clip_path: str,
    batch_items: dict,
    video_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    frames_list: list[torch.Tensor] = []
    for clip_uid, clip_start_sec, clip_end_sec in zip(
        batch_items["clip_uid"],
        batch_items["clip_start_sec"],
        batch_items["clip_end_sec"],
    ):
        # Load the video corresponding to the clip
        video = video_path_handler.video_from_path(
            os.path.join(clip_path, clip_uid + ".mp4")
        )

        # extract the frames corresponding to the action
        frames = video.get_clip(clip_start_sec, clip_end_sec)["video"]

        if video_transform is not None:
            frames = video_transform(frames)
        # put the frames through the processor
        frames = image_processor(frames.permute(1, 0, 2, 3), return_tensors="pt")[
            "pixel_values"
        ].permute(1, 0, 2, 3)
        frames_list.append(frames)
    del batch_items["clip_uid"]
    del batch_items["clip_start_sec"]
    del batch_items["clip_end_sec"]

    # (batch, channel, time, height, width)
    batch_items["pixel_values"] = torch.stack(frames_list)

    return batch_items


# NOTE: We can't use 3.10's new X|Y syntax b/c HfArgumentParser doesn't support it.
# https://github.com/huggingface/transformers/issues/20249
@dataclass
class ModelArguments:
    model_name_or_path: str


@dataclass
class DataArguments:
    annotation_path: str
    clip_path: str
    instruct_tuned: bool


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    val_set_size: float = field(default=0.1)


class DataCollatorForVideoLanguageModeling(
    transformers.DataCollatorForLanguageModeling
):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        collated["pixel_values"] = pixel_values
        return collated


class DataCollatorForVideoSeq2Seq(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )
        collated = super().__call__(features, return_tensors=return_tensors)
        collated["pixel_values"] = pixel_values
        return collated


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Don't remove "unused columns" such as clip-related columns
    training_args.remove_unused_columns = False

    processor = transformers.Blip2Processor.from_pretrained(
        model_args.model_name_or_path
    )
    model = VideoBlip2ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=False if is_deepspeed_zero3_enabled() else True,
    )
    # freeze everything except for qformer
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    # we need to enable input require grads since the vision model (the first layer) is
    # frozen.
    model.enable_input_require_grads()

    with training_args.main_process_first(desc="load and preprocess dataset"):
        dataset = (
            load_ego4d_fho_clip_dataset(data_args.annotation_path)
            .filter(  # filter out rejected, invalid and non-C actions
                lambda is_rejected, is_valid_action, narration_text: not is_rejected
                and is_valid_action
                and C_REGEX.match(narration_text),
                input_columns=["is_rejected", "is_valid_action", "narration_text"],
            )
            .remove_columns(  # remove unused columns
                [
                    "warnings",
                    "uid",
                    "start_sec",
                    "end_sec",
                    "start_frame",
                    "end_frame",
                    "is_valid_action",
                    "is_partial",
                    "clip_start_frame",
                    "clip_end_frame",
                    "narration_timestamp_sec",
                    "clip_narration_timestamp_sec",
                    "narration_annotation_uid",
                    "structured_verb",
                    "freeform_verb",
                    "state_transition",
                    "critical_frames",
                    "clip_critical_frames",
                    "frames",
                    "is_rejected",
                    "is_invalid_annotation",
                    "reject_reason",
                    "stage",
                ]
            )
            .map(  # clean narration_text
                clean_narration_text,
                input_columns="narration_text",
                remove_columns="narration_text",
            )
            .map(  # add prompt as a column
                partial(add_prompt_column, instruct_tuned=data_args.instruct_tuned)
            )
            .map(  # generate inputs using the prompt and cleaned_narration_text
                partial(
                    generate_inputs,
                    use_decoder_only_lm=model.config.use_decoder_only_language_model,
                ),
                remove_columns=["prompt", "cleaned_narration_text"],
            )
            .map(  # tokenize text inputs
                partial(
                    batch_tokenize,
                    processor.tokenizer,
                    use_decoder_only_lm=model.config.use_decoder_only_language_model,
                ),
                batched=True,
                remove_columns="input",
            )
        )
        if training_args.val_set_size > 0:
            logging.warning("Splitting train and validation datasets")
            train_val = dataset.train_test_split(
                test_size=training_args.val_set_size, shuffle=True, seed=42
            )
            train_data = train_val["train"]
            val_data = train_val["test"]

            # set video transform functions
            video_path_handler = VideoPathHandler()
            train_transform = Compose([UniformTemporalSubsample(8)])
            val_transform = Compose([UniformTemporalSubsample(8)])
            train_data.set_transform(
                partial(
                    extract_frames,
                    video_path_handler,
                    processor.image_processor,
                    data_args.clip_path,
                    video_transform=train_transform,
                )
            )
            val_data.set_transform(
                partial(
                    extract_frames,
                    video_path_handler,
                    processor.image_processor,
                    data_args.clip_path,
                    video_transform=val_transform,
                )
            )

            # Load the best model at the end so we can save it
            training_args.load_best_model_at_end = True
        else:
            logging.warning("No validation set")
            train_data = dataset
            val_data = None

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForVideoLanguageModeling(
            processor.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        )
        if model.config.use_decoder_only_language_model
        else DataCollatorForVideoSeq2Seq(
            processor.tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        ),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
