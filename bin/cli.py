from pathlib import Path
import torch
from torch import optim

from ser.model import Net
from ser.transform import normalise
from ser.data import get_training_dataloader, get_validation_dataloader
from ser.train import begin_training

import typer

import json
import logging
from datetime import datetime
from git import Repo


main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        10, "-t", "--epochs", help="Number of time steps to train."
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size for training."
    ),
    learning_rate: float = typer.Option(
        0.01, "-r", "--learning-rate", help="Learning rate for training." 
    )
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(f'./logs/{name}-{datetime.now()}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Running experiment {name} on {device} device...")
    logger.info(f"Config: epochs = {epochs}, batch_size = {batch_size}, learning rate = {learning_rate}. ")
    # save the parameters!

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = normalise()

    # dataloaders
    training_dataloader = get_training_dataloader(ts, batch_size)
    validation_dataloader = get_validation_dataloader(ts, batch_size)

    # train
    begin_training(model, training_dataloader, validation_dataloader, epochs, optimizer, device, logger)
    Path(f"run/{name}").mkdir(parents=True, exist_ok=True)
    completion_time = datetime.now()
    repo = Repo(search_parent_directories=True)
    sha = repo.head.commit.hexsha
    params_dict = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'commit_sha': sha,
    }
    with open(f'./run/{name}/{completion_time}.json', 'w') as f:
        json.dump(params_dict, f)
    torch.save(model, f'run/{name}/{completion_time}')
@main.command()
def infer():
    print("This is where the inference code will go")
