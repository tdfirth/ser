from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import typer

from ser.train import my_train

main = typer.Typer()

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
):
    my_train(name)
    


@main.command()
def infer():
    print("This is where the inference code will go")
