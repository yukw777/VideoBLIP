import pytest

from ego4d_dataset import Ego4dFHOClipDataset


@pytest.mark.parametrize(
    "narration_text,cleaned",
    [
        ("#C C drops a plate", "The camera wearer drops a plate"),
        ("#C C drops a plate ", "The camera wearer drops a plate"),
        ("#c C drops a plate", "The camera wearer drops a plate"),
        ("#C c drops a plate", "The camera wearer drops a plate"),
    ],
)
def test_ego4d_fho_clip_dataset_clean_narration_text(
    narration_text: str, cleaned: str
) -> None:
    assert Ego4dFHOClipDataset._clean_narration_text(narration_text) == cleaned
