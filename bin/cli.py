from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.train import train

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#or ser run train params

@main.command() #ser params "name"
def params(name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."),
        model_setting: str = typer.Option(
        ..., "-s", "--setting", help="'train' or 'run' the model."),
        epochs: int = typer.Option(
            2, "e", "--epochs", help="number of epochs to train the model for."
        ),
        batch_size: int = typer.Option(
            1000, "b", "--batch_size", help="batch size."
        ),
        learning_rate: float = typer.Option(
            0.01, "lr", "--learning_rate", help="learning rate."
        )
        ):

    if str.lower(model_setting) == 'infer':
        train(name, epochs, batch_size, learning_rate)
    elif str.lower(model_setting) == 'run':
        infer()
    else:
        print("please specify whether you wnat to train or run the model")