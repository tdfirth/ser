from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataloader(root, train, transform, batch_size):
    return DataLoader(
        datasets.MNIST(root=root, download=True, train=train, transform=transform),
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,)

