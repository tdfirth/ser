from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer
from ser.model import modeldevice
from ser.transforms import torchtransforms
from ser.data import dataloader
from ser.train import modeltrain

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning_rate", help="Learning rate."
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!

    # load model
    model = modeldevice(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = torchtransforms()

    # dataloaders
    training_dataloader = dataloader("../data", True, ts, batch_size)
    validation_dataloader = dataloader(DATA_DIR, False, ts, batch_size)

    # train
    modeltrain(epochs, device, model, optimizer, training_dataloader, validation_dataloader)



@main.command()
def infer():
    print("This is where the inference code will go")
