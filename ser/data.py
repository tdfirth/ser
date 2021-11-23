from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# torch transforms
ts = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def train_loader(batch_size, mnist=True, ts=ts):
    if mnist == True:
        out = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts),
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )
        return out
    else:
        return "other datsets not supported"


def validation_loader(batch_size, mnist=True, ts=ts):
    if mnist == True:
        out = DataLoader(
            datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
        )
        return out
    else:
        return "other datasets not supported"
