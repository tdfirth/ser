import numpy as np
import pytest
import torch

from ser.transforms import flip, transforms


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])),
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),
    ],
)
def test_flip(array, expected):
    ts = transforms(flip)
    assert (ts(array) == torch.Tensor(expected)).all()
