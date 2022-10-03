from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import torch
import git
import json

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize
from ser.selectimage import select_image
from ser.runinference import inference

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        5, "-e", "--epochs", help="Number of epochs to run for."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for dataloader."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning-rate", help="Learning rate for the model."
    ),
):
    """Run the training algorithm."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # wraps the passed in parameters
    params = Params(name, epochs, batch_size, learning_rate, sha)

    # setup device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup run
    fmt = "%Y-%m-%dT%H-%M"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = RESULTS_DIR / name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # Save parameters for the run
    save_params(run_path, params)

    # Train!
    run_train(
        run_path,
        params,
        train_dataloader(params.batch_size, transforms(normalize)),
        val_dataloader(params.batch_size, transforms(normalize)),
        device,
    )


@main.command()
def infer(
    run_path: Path = typer.Option(
        ..., "-r", "--run_path", help="Run path."
    ),
    label: int = typer.Option(
        3, "-l", "--label", help="Number of label."
    ),
):

    run_path = RESULTS_DIR / run_path

    # TODO load the parameters from the run_path so we can print them out!
    with open(run_path / "params.json", 'r') as f:
        params = json.load(f)
    print('Experiment settings:')
    print(params)

    # select image to run inference for
    images = select_image(label)

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    inference(model, images)