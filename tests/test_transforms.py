from ser.transforms import flip
from torchvision import transforms as torch_transforms
import torch


def test_flip():
    # original is 
    # 1 2
    # 3 4

    # flipped is
    # 4 3
    # 2 1

    original = torch.tensor([[[1,2],[3,4]]])
    flipped = torch.tensor([[[4,3],[2,1]]])
    assert torch.equal(flip()(original), flipped)


def test_flip_samedigit():
    # original is 
    # 0 0
    # 0 0

    # flipped is
    # 0 0
    # 0 0

    original = torch.tensor([[[0,0],[0,0]]])
    flipped = torch.tensor([[[0,0],[0,0]]])
    assert torch.equal(flip()(original), flipped)