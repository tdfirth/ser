import torch

from ser.transforms import normalize, flip


def test_normalize():
    """
    Input tensor has shape (C, H, W) for gray scale images (channel, height, width).
    https://pytorch.org/vision/stable/transforms.html?highlight=normalize#torchvision.transforms.functional.normalize
    We can use this information to create input data with the right interface.
    Because our test data is so small, we are also able to easily work out the correct answer.
    """
    # If the input is all 1s, then what should the answer be?
    # (1 - 0.5) / 0.5 = 1... so we should see that the input is unchanged.
    assert torch.equal(
        normalize()(torch.FloatTensor([[[1, 1, 1], [1, 1, 1]]])),
        torch.FloatTensor([[[1, 1, 1], [1, 1, 1]]]),
    )

    # If the input is something else, we should see the appropriate transformation applied.
    assert torch.equal(
        normalize()(torch.FloatTensor([[[1, 2, 3], [3, 2, 1]]])),
        torch.FloatTensor([[[1, 3, 5], [5, 3, 1]]]),
    )


def test_flip():
    """
    Flip is meant to flip an image both horizontally vertically.
    I.e. it flips it along a diagonal axis, turning a 6 into a 9.
    A good example is:
      1 0 1
      0 1 0
      0 0 1
    We expect that to be flipped into:
      1 0 0
      0 1 0
      1 0 1
    So lets test that!
    """
    input = torch.FloatTensor([[[1, 0, 1], [0, 1, 0], [0, 0, 1]]])
    output = torch.FloatTensor([[[1, 0, 0], [0, 1, 0], [1, 0, 1]]])
    assert torch.equal(flip()(input), output)
