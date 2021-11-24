from datetime import datetime
from pathlib import Path

import typer
import torch
import git

from ser.train import train as run_train
from ser.train import instance_predict
from ser.constants import PROJECT_ROOT, RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader, get_one_test_instance
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize
from ser.ascii_art import generate_ascii_art

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
    run_path: str = typer.Argument(...,
                                   help='path in the results dir for the model to evaluate'),
    label: int = typer.Option(
        6, "-l", "--label", help="ground truth label for model to infer"),

):

    # TODO load the parameters from the run_path so we can print them out!
    run_path = PROJECT_ROOT / run_path
    params = load_params(run_path, verbose=True)
    # select image to run inference for
    images = get_one_test_instance(label, transforms(normalize))
    # load the model
    model = torch.load(run_path / "model.pt")
    # run inference
    pixels, pred, certainty = instance_predict(model, images)
    print(generate_ascii_art(pixels))
    print(f"This is a {pred} with certainty {certainty}.")
