from unittest.mock import Mock

import pytest
import torch
from transformers import BatchEncoding

from video_blip2.data.utils import clean_narration_text, generate_input_ids_and_labels


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


@pytest.mark.parametrize(
    "decoder_only_lm,tokenizer,expected",
    [
        (
            True,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1]}),
                ],
                eos_token_id=100,
            ),
            BatchEncoding(
                data={
                    "input_ids": torch.tensor([1, 2, 3, 4, 4, 3, 2, 1, 100]),
                    "labels": torch.tensor([-100, -100, -100, -100, 4, 3, 2, 1, 100]),
                }
            ),
        ),
        (
            False,
            Mock(
                side_effect=[
                    BatchEncoding(data={"input_ids": [1, 2, 3, 4, 100]}),
                    BatchEncoding(data={"input_ids": [4, 3, 2, 1, 100]}),
                ]
            ),
            BatchEncoding(
                data={
                    "input_ids": torch.tensor([1, 2, 3, 4, 100]),
                    "labels": torch.tensor([4, 3, 2, 1, 100]),
                }
            ),
        ),
    ],
)
def test_generate_input_ids_and_labels(
    decoder_only_lm: bool, tokenizer, expected: BatchEncoding
) -> None:
    results = generate_input_ids_and_labels(tokenizer, "", "", decoder_only_lm)
    assert results.keys() == expected.keys()
    assert results.input_ids.equal(expected.input_ids)
    assert results.labels.equal(expected.labels)
