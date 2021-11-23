from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from ser import model, data, trainval

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(..., "-e", "--epochs", help="Number of epochs."),
    batch_size: int = typer.Option(..., "-b", "--batch_size", help="Batch size."),
    learning_rate: float = typer.Option(
        ..., "-l", "--learning_rate", help="Learning rate."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!

    # load model
    main_model = model.Net().to(device)

    # setup params
    optimizer = optim.Adam(main_model.parameters(), lr=learning_rate)

    # dataloaders
    training_dataloader = data.train_loader(batch_size)
    validation_dataloader = data.validation_loader(batch_size)

    trainval.train(main_model, epochs, training_dataloader, device, optimizer)
    # validate
    trainval.validate(main_model, validation_dataloader, device)


@main.command()
def infer():
    print("This is where the inference code will go")
