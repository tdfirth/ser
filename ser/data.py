from torch.utils.data import DataLoader
from torchvision import datasets

from ser.constants import Transform, DATA_DIR


def get_data(
    transform: Transform, batch_size: int, train: bool = True, shuffle: bool = True
):
    return DataLoader(
        datasets.MNIST(
            root=str(DATA_DIR), download=True, train=train, transform=transform
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )

# Returns dataloader with mnist validation data.
# Note: normally this would be real 'test' or future data
def test_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
