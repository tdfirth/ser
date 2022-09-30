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

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


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
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_dataloader, validation_dataloader = dataloader(batch_size, DATA_DIR)

    # train
    trainer(epochs,training_dataloader, validation_dataloader,device,model,optimizer)

@main.command()
def infer():
    print("This is where the inference code will go")
