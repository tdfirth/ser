from torchvision import transforms as torch_transforms


def transforms(*stages):
    return torch_transforms.Compose(
        [
            torch_transforms.ToTensor(),
            *(stage() for stage in stages),
        ]
    )


def normalize():
    """
    Normalize a tensor to have a mean of 0.5 and a std dev of 0.5
    """
    return torch_transforms.Normalize((0.5,), (0.5,))
