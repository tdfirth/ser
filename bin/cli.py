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
        ..., "-n", "--setting", help="'train' or 'run' the model.")):
    if str.lower(model_setting) == 'infer':
        train()
    elif str.lower(model_setting) == 'run':
        infer()
    else:
        print("please ")
