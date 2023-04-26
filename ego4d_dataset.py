import json
import logging
import re

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

INSTR_PROMPT = "What is the camera wearer doing?"


class Ego4dFHOClipDataset(Dataset):
    """Ego4d v2 video clip dataset for the fho benchmark.

    This dataset is for the clips, not the full scale videos.
    """

    C_REGEX = re.compile(r"^\#C C", re.IGNORECASE)

    def __init__(self, annotation_path: str, prompt: str = INSTR_PROMPT) -> None:
        """
        :param annotation_path: path to fho_main.json
        :param prompt: input prompt
        """
        self._prompt = prompt
        with open(annotation_path) as f:
            annotations = json.load(f)
        self._actions: list[dict] = []
        num_rejected_actions = 0
        num_invalid_actions = 0
        num_o_actions = 0
        for video in annotations["videos"]:
            for interval in video["annotated_intervals"]:
                for action in interval["narrated_actions"]:
                    if action["is_rejected"]:
                        num_rejected_actions += 1
                        continue
                    if not action["is_valid_action"]:
                        num_invalid_actions += 1
                        continue
                    if not self.C_REGEX.match(action["narration_text"]):
                        num_o_actions += 1
                        continue
                    self._actions.append(
                        {
                            "input": self._prompt,
                            "labels": self._clean_narration_text(
                                action["narration_text"]
                            ),
                        }
                    )
        logger.info(f"# of filtered rejected actions: {num_rejected_actions}")
        logger.info(f"# of filtered invalid actions: {num_invalid_actions}")
        logger.info(f"# of filtered number of #O actions: {num_o_actions}")

    @classmethod
    def _clean_narration_text(cls, narration_text: str) -> str:
        # strip it first
        cleaned = narration_text.strip()

        # replace "#C C" with "The camera wearer"
        cleaned = re.sub(cls.C_REGEX, "The camera wearer", narration_text.strip())

        return cleaned.strip()

    def __getitem__(self, index: int) -> dict:
        return self._actions[index]

    def __len__(self) -> int:
        return len(self._actions)
