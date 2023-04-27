import json
import re

from datasets import Dataset
from transformers import Blip2Processor

INSTR_PROMPT = "What is the camera wearer doing?"


class Ego4dFHOClipDataset(Dataset):
    """Ego4d v2 video clip dataset for the fho benchmark.

    This dataset is for the clips, not the full scale videos.
    """

    C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)

    def __init__(
        self,
        annotation_path: str,
        processor: Blip2Processor,
        use_decoder_only_language_model: bool,
    ) -> None:
        """
        :param annotation_path: path to fho_main.json
        :param processor: Blip2Processor
        :param use_decoder_only_language_model:
            whether we're using decoder only language model
        """
        self.processor = processor
        self.use_decoder_only_language_model = use_decoder_only_language_model

        with open(annotation_path) as f:
            annotations = json.load(f)

        # create a dataset out of all the narrated actions
        self._dataset = Dataset.from_list(
            [
                {"clip_uid": interval["clip_uid"], **action}
                for video in annotations["videos"]
                for interval in video["annotated_intervals"]
                for action in interval["narrated_actions"]
            ]
        )

        # filter out rejected, invalid and non-C actions
        self._dataset = self._dataset.filter(
            lambda is_rejected, is_valid_action, narration_text: not is_rejected
            and is_valid_action
            and self.C_REGEX.match(narration_text),
            input_columns=["is_rejected", "is_valid_action", "narration_text"],
        )

        # remove unused columns
        self._dataset = self._dataset.remove_columns(
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

        # preprocess the texts
        self._dataset = self._dataset.map(
            self._batch_preprocess_texts, batched=True, remove_columns="narration_text"
        )

    def _batch_preprocess_texts(self, batch_examples) -> dict[str, list[int]]:
        """Preprocess text inputs. If using a decoder only language model, we
        concatenate the prompt with "narration_text" into a single input. For
        seq2seq language models, we put the prompt as the input and
        narration_text as the label.

        Note that we don't need to do any right shifting here since
        that's handled by the model. For example,
        T5ForconditionalGeneration right shifts the label using the pad
        token, while GPT2LMHeadModel removes the last token, i.e., eos,
        from the input and removes the first token from the label, which
        is set to be the same as the input by
        DataCollatorForLanguageModeling.
        """
        # NOTE: since examples will be batched by the collator, we don't need to
        # generate attention masks here.
        if self.use_decoder_only_language_model:
            # tokenize and append eos
            tokenized = self.processor.tokenizer(
                [
                    INSTR_PROMPT + " " + self._clean_narration_text(narration_text)
                    for narration_text in batch_examples["narration_text"]
                ],
                return_attention_mask=False,
            )
            for input_ids in tokenized.input_ids:
                input_ids.append(self.processor.tokenizer.eos_token_id)
            return tokenized
        else:
            # eos is automatically appended by the tokenizer
            return {
                "input_ids": self.processor.tokenizer(
                    [
                        INSTR_PROMPT
                        for _ in range(len(batch_examples["narration_text"]))
                    ],
                    return_attention_mask=False,
                ).input_ids,
                "labels": self.processor.tokenizer(
                    [
                        self._clean_narration_text(narration_text)
                        for narration_text in batch_examples["narration_text"]
                    ],
                    return_attention_mask=False,
                ).input_ids,
            }

    @classmethod
    def _clean_narration_text(cls, narration_text: str) -> str:
        # strip it first
        cleaned = narration_text.strip()

        # replace "#C C" with "The camera wearer"
        cleaned = re.sub(cls.C_REGEX, "The camera wearer", narration_text.strip())

        return cleaned.strip()

    def __getitem__(self, index: int) -> dict:
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)
