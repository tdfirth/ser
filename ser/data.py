from torch.utils.data import DataLoader
from torchvision import datasets

from ser.constants import DATA_DIR


def train_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=True, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)


def val_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)


# Returns dataloader with mnist validation data.
# Note: normally this would be real 'test' or future data
def test_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
