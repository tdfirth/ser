from pathlib import Path
import torch
from torch import optim

import typer

# importing model
from ser.model import Net
# importing data loaders
from ser.data import dataloader
# importing trainer
from ser.train import trainer

import json

from datetime import date
import time

from dataclasses import dataclass

from ser.git_saver import git_hash

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@dataclass
class hyperparams:
    epochs: int

    batch_size: int

    learning_rate: float

    git_hash: str

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),

    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs."
    ),

    batch_size: int = typer.Option(
        ..., "-b", "--batch", help="Batch size."
    ),

    learning_rate: float = typer.Option(
        ..., "-l", "--learning", help ="Learning rate."
    )
):
    experiment_hps = hyperparams(epochs,batch_size,learning_rate,git_hash())

    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open(f'{name}_dmy{timestr}_hyperperams.json', "a") as f:
        json.dump({"experiment_name": name, "datetime": timestr, "epochs": experiment_hps.epochs, "batch_size": experiment_hps.batch_size, "learning_rate": experiment_hps.learning_rate, "git_hash": experiment_hps.git_hash}, f)

    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=experiment_hps.learning_rate)

    training_dataloader, validation_dataloader = dataloader(experiment_hps.batch_size, DATA_DIR)

    # train
    trainer(experiment_hps.epochs,training_dataloader, validation_dataloader,device,model,optimizer,name)

@main.command()
def infer():
    print("This is where the inference code will go")
