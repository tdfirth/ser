from ser.data import test_dataloader
from ser.transforms import transforms, normalize


def select_image(label):
# select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return(dataloader, images, labels)