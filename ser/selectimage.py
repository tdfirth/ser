from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.transforms import transforms, normalize


def select_image(label):
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images