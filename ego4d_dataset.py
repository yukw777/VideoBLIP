from typing import Any

from torch.utils.data import Dataset


class Ego4dFHOClipDataset(Dataset):
    """Ego4d v2 video clip dataset for the fho benchmark.

    This dataset is for the clips, not the full scale videos.
    """

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return 0
