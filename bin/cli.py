import json
import os
import torch
import typer
from pathlib import Path
from torch import optim
from typing import Optional, List

from bin.validators import transform_callback
from ser.constants import OUTPUTS_DIR
from ser.data import get_data
from ser.infer import infer_label, select_test_image
from ser.models import Net, Parameters, TrainingModel, Data, get_file_path
from ser.outputs import get_params_in_dir
from ser.train import train_model
from ser.transforms import normalize, transforms
from utils import get_unique_id, write_dataclass_dict, load_object_from_json

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
    write_dataclass_dict(class_object=parameters, file_path=get_file_path(parameters))

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
    file_path: Path = typer.Option(
        ..., "-p", "--path", help="Path to run from which you want to infer."
    ),
    predict_number: int = typer.Option(
        6, "-l", "--label", help="Label of image to show to the model"
    ),
    transform_list: List[str] = typer.Option(
        [],
        "-tl",
        "--transform-list",
        help="List of transforms to input",
        callback=transform_callback,
    ),
):

    transform_list = [normalize, *transform_list]

    parameters = load_object_from_json(Parameters, file_path / "parameters.json")

    model = torch.load(file_path / "model.pt")

    print(f"\nRunning inference")
    print(parameters)
    print(f"The image you have asked to classify is a {predict_number}.")
    print(f"Here's what was fed to the model after transformations were applied:")

    image = select_test_image(predict_number, transform_list)
    infer_label(model, image)


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
