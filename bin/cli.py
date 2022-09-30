from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from ser.model import modelsetup
from ser.transforms import transform
from ser.data import load_train_data, load_val_data
from ser.train import training, validation

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", help="Number of epochs to run."
    ),
    batch_size: int = typer.Option(
        ..., "-bs", help="Batch size to use."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", help="Learning rate to use."
    )
):
    print(f"Running experiment {name}")
    
    device, model, optimizer = modelsetup(learning_rate)

    # torch transforms
    ts = transform()

    # dataloaders
    training_dataloader = load_train_data(batch_size=batch_size, ts=ts)

    validation_dataloader = load_val_data(batch_size=batch_size, ts=ts)

    # train
    for epoch in range(epochs):
        training(epoch, training_dataloader, model, device, optimizer)
        validation(epoch, validation_dataloader, model, device)

@main.command()
def infer():
    print("This is where the inference code will go")
