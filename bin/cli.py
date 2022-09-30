from pathlib import Path
from typing import Any
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import json
import git

import typer
from ser.model import modeldevice
from ser.transforms import torchtransforms
from ser.data import dataloader
from ser.train import modeltrain
from dataclasses import dataclass, asdict

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        0.01, "-l", "--learning_rate", help="Learning rate."
    )
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datetime_start = str(datetime.now())

    # create folders for outputs - Hyperparameters, Models ...
    path_params = str(PROJECT_ROOT) + "/Outputs/Hyperparameters/"
    path_model = str(PROJECT_ROOT) + "/Outputs/Models/"
    path_results = str(PROJECT_ROOT) + "/Outputs/Validation_accuracy/"
    Path(path_params).mkdir(parents=True, exist_ok=True)
    Path(path_model).mkdir(parents=True, exist_ok=True)
    Path(path_results).mkdir(parents=True, exist_ok=True)

    # get the git commit hash
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.git.rev_parse("HEAD")

    # print warning message when there are uncommitted changes
    if repo.is_dirty():
        print("The repository has uncommitted changes.")

    # save the parameters!
    path_params_model = path_params + name + "_" + datetime_start + ".json"
    #params = {"Name": name, "Epochs number": epochs, 
    #"Batch size": batch_size, "Learning rate": learning_rate,
    #"Datetime": datetime_start}
    params = Parameters(name, epochs, batch_size, learning_rate, datetime_start, commit_hash)
    params_dict = asdict(params)
    with open(path_params_model, "w") as fp:
        json.dump(params_dict,fp) 

    # load model
    model = modeldevice(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = torchtransforms()

    # dataloaders
    training_dataloader = dataloader("../data", True, ts, batch_size)
    validation_dataloader = dataloader(DATA_DIR, False, ts, batch_size)

    # train
    model, results = modeltrain(epochs, device, model, optimizer, training_dataloader, validation_dataloader)

    # save the model
    path_train_model = path_model + name + "_" + datetime_start + ".pth"
    torch.save(model, path_train_model)

    # save the highest validation accuracy for all epochs 
    path_accuracy = path_results + name + "_" + datetime_start + ".json"
    with open(path_accuracy, "w") as fp:
        json.dump(results,fp) 


@main.command()
def infer():
    print("This is where the inference code will go")

@dataclass
class Parameters:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    datetime: datetime
    git_commit_hash: Any
