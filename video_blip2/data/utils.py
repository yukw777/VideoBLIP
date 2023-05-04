import re

from video_blip2.data.ego4d import Ego4dFHOMainDataset


def clean_narration_text(narration_text: str) -> str:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    return re.sub(Ego4dFHOMainDataset.C_REGEX, "The camera wearer", cleaned)
