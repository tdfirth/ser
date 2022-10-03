from datetime import datetime
from pathlib import Path

import typer
import torch
import git
import json
import os

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize
from ser.select import select_image
from ser.run import run_inference
from ser.artwork import generate_ascii_art, pixel_to_char

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
    experiment: str = typer.Option(
        "experiment1", "-x", "--experiment", help="Name of experiment you wish to perform inference from."
    ),
    run: str = typer.Option(
        "2022-10-03T10-27", "-r", "--run", help="Date and time of run you wish to perform inference from, in format YYYY-MM-DDThh-mm."
    ),
):
    run_path = Path("./results",experiment,run)
    label = 6

    # TODO load the parameters from the run_path so we can print them out!

    # select image to run inference for
    select_image(label)

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    run_inference(model, label)

    #Print summary of experiment name and hyperparameters
    run_params = open(Path(run_path,'params.json'))
    run_params_dict = json.load(run_params)
    print("Experiment Name: "+run_params_dict["name"]+
    "\nEpochs: "+str(run_params_dict["epochs"])+
    "\nBatch Size: "+str(run_params_dict["batch_size"])+
    "\nLearning Rate: "+str(run_params_dict["learning_rate"]))

@main.command()
def summarize_runs():
    results_path = Path("./results")

    # print out all files in results directory
    for (root, dirs, file) in os.walk(results_path):
        for f in dirs:
            print(f)