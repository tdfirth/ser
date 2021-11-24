import json
from pathlib import Path

import torch
import typer
from torch import optim

from ser.data import get_data
from ser.infer import infer_label
from ser.models import Net, Parameters, TrainingModel, Data
from ser.train import train_model
from ser.transforms import normalize, transforms
from utils import get_unique_id

main = typer.Typer()


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
    parameters = Parameters(get_unique_id(), name, epochs, batch_size, learning_rate)

    print(f"Running experiment {name}")

    training_model = TrainingModel(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        _model=Net(),
        _optimizer=optim.Adam,
        parameters=parameters,
    )

    # torch transforms
    ts = transforms(normalize)

    # dataloaders
    data = Data.from_inputs(ts, parameters)

    train_model(parameters, data, training_model)


@main.command()
def infer(
    file_path: str = typer.Option(
        ..., "-fp", "--file-path", help="Name of file path to load"
    ),
    predict_number: int = typer.Option(
        6, "-n", "--predict-number", help="Number to infer"
    ),
):
    run_path = Path(file_path)

    with open(run_path / "parameters.json") as f:
        parameters = Parameters(**json.load(f))

    print(parameters)

    # select image to run inference for
    dataloader = get_data(transform=transforms(normalize), batch_size=1, train=False)
    infer_label(dataloader, run_path, predict_number)



def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
