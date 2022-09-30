#input data specified here
#MINST DATA
from torch.utils.data import DataLoader
from torchvision import datasets

from ser.transforms import transform

#load_data
def load_data():
    dataset=datasets.MNIST(root="../data", download=True, train=True, transform=transform()) 
    return dataset

# dataloaders
def dataloaders(batch_size):
    dataset = load_data()
    training_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return training_dataloader, validation_dataloader