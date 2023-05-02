import json
import os
import random
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from pytorchvideo.data import ClipSampler, LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipInfo

C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)


class RandomNarratedActionClipSampler(ClipSampler):
    def __init__(self) -> None:
        """The vast majority of narrated actions are 8 seconds long, and none
        are longer.

        So let's just sample 8-second clips.
        """
        super().__init__(8)
        self.shuffled_clip_indices: list[int] | None = None

    def __call__(
        self,
        last_clip_time: float | Fraction,
        video_duration: float | Fraction,
        annotation: dict[str, Any],
    ) -> ClipInfo:
        """Draw a random clip for a narrated action.

        :param last_clip_time: unused
        :param video_duration: unused
        :param annotation: narrated action data.
            See https://ego4d-data.org/docs/data/annotations-schemas/ for more details.
        """
        if self.shuffled_clip_indices is None:
            # first time sampling from this video, so create a shuffled list
            self.shuffled_clip_indices = list(
                range(len(annotation["narrated_actions"]))
            )
            random.shuffle(self.shuffled_clip_indices)

        clip_index = self.shuffled_clip_indices[self._current_clip_index]
        narrated_action = annotation["narrated_actions"][clip_index]
        self._current_clip_index += 1

        is_last_clip = False
        if self._current_clip_index == len(self.shuffled_clip_indices):
            is_last_clip = True

        # sample a clip 8 seconds around narration_time_sec
        # if narration_time_sec is less than 4 seconds, we start from 0
        clip_start_sec = max(
            Fraction(narrated_action["narration_timestamp_sec"])
            - self._clip_duration / 2,
            0,
        )

        # add 8 seconds to clip_start_sec
        # if clip_end_sec goes over the video duration, adjust clip_start_sec
        clip_end_sec = clip_start_sec + self._clip_duration
        video_duration_sec = Fraction(annotation["video_metadata"]["duration_sec"])
        if clip_end_sec > video_duration_sec:
            clip_end_sec = video_duration_sec
            clip_start_sec = clip_end_sec - self._clip_duration

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_end_sec,
            clip_index,
            0,
            is_last_clip,
        )

    def reset(self) -> None:
        self._current_clip_index = 0
        self.shuffled_clip_indices = None


class Ego4dFHOMainDataset(LabeledVideoDataset):
    def __init__(
        self,
        annotation_path: str,
        video_path: str,
        transform: Callable[[dict], Any] | None = None,
    ) -> None:
        with open(annotation_path) as f:
            annotations = json.load(f)

        def _transform(item: dict) -> Any:
            """The first transform function that formats `narrated_actions`."""
            narrated_actions = item.pop("narrated_actions")
            item.update(narrated_actions[item["clip_index"]])
            if transform is not None:
                item = transform(item)
            return item

        super().__init__(
            [
                (
                    os.path.join(video_path, video["video_uid"] + ".mp4"),
                    {
                        "narrated_actions": [
                            {
                                "start_sec": action["start_sec"],
                                "end_sec": action["end_sec"],
                                "narration_text": action["narration_text"],
                            }
                            for interval in video["annotated_intervals"]
                            for action in interval["narrated_actions"]
                            if not action["is_rejected"]
                            and action["is_valid_action"]
                            and C_REGEX.match(action["narration_text"])
                        ]
                    },
                )
                for video in annotations["videos"]
            ],
            RandomNarratedActionClipSampler(),
            transform=_transform,
            decode_audio=False,
        )
