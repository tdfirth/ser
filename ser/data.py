from torch.utils.data import DataLoader
from torchvision import datasets

def get_training_dataloader(ts, batch_size):
    dl = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return dl
def get_validation_dataloader(ts, batch_size):
    dl = DataLoader(
        datasets.MNIST(root="../data", download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return dl
