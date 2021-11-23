from torchvision import transforms


def get_transforms():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
