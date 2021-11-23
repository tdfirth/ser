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
