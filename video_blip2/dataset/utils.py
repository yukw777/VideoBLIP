import re

from video_blip2.dataset.ego4d import Ego4dFHOMainDataset


def clean_narration_text(narration_text: str) -> dict[str, str]:
    # strip it first
    cleaned = narration_text.strip()

    # replace "#C C" with "The camera wearer"
    cleaned = re.sub(Ego4dFHOMainDataset.C_REGEX, "The camera wearer", cleaned)

    return {"cleaned_narration_text": cleaned}
