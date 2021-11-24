import json
import os
from pathlib import Path
from typing import Optional

import torch
import typer
from torch import optim

from ser.constants import OUTPUTS_DIR
from ser.data import get_data
from ser.infer import infer_label
from ser.models import Net, Parameters, TrainingModel, Data
from ser.outputs import get_params_in_dir
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
    file_path = Path(file_path)

    with open(file_path / "parameters.json") as f:
        parameters = Parameters(**json.load(f))

    print(parameters)

    # select image to run inference for
    dataloader = get_data(transform=transforms(normalize), batch_size=1, train=False)
    infer_label(dataloader, file_path, predict_number)


@main.command()
def get_experiments(
    root_dir: str = typer.Option(
        OUTPUTS_DIR, "-rd", "--root-dir", help="Name of root directory to load from"
    ),
    experiment_filter: Optional[str] = typer.Option(
        None, "-ef", "--experiment", help="Name of particular experiment to filter on"
    ),
):

    root_dir = Path(root_dir)

    if not experiment_filter:
        for experiment_name in os.listdir(root_dir):
            get_params_in_dir(root_dir / experiment_name)

    else:
        get_params_in_dir(root_dir / experiment_filter)
