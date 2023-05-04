import pytest

from video_blip2.data.utils import clean_narration_text


@pytest.mark.parametrize(
    "narration_text,cleaned",
    [
        ("#C C drops a plate", "The camera wearer drops a plate"),
        ("#C C drops a plate ", "The camera wearer drops a plate"),
        ("#c C drops a plate", "The camera wearer drops a plate"),
        ("#C c drops a plate", "The camera wearer drops a plate"),
    ],
)
def test_clean_narration_text(narration_text: str, cleaned: str) -> None:
    assert clean_narration_text(narration_text) == cleaned
