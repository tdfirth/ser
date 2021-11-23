from pathlib import Path

import torch
import typer
from torch import optim

from ser.models import Net, Parameters, TrainingModel, Data
from ser.train import train_model
from ser.tranforms import get_transforms

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs to train over"
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size to train with"
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning-rate", help="Learning rate to train with"
    ),
):
    parameters = Parameters(name, epochs, batch_size, learning_rate)

    print(f"Running experiment {name}")

    training_model = TrainingModel(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        _model=Net(),
        _optimizer=optim.Adam,
        parameters=parameters,
    )

    # torch transforms
    ts = get_transforms()

    # dataloaders
    data = Data.from_inputs(ts, parameters)

    train_model(parameters, data, training_model)


@main.command()
def infer():
    print("This is where the inference code will go")
