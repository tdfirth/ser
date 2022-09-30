from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from ser.model import modelsetup
from ser.transforms import transform
from ser.data import load_train_data, load_val_data

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", help="Number of epochs to run."
    ),
    batch_size: int = typer.Option(
        ..., "-bs", help="Batch size to use."
    ),
    learning_rate: float = typer.Option(
        ..., "-lr", help="Learning rate to use."
    )
):
    print(f"Running experiment {name}")
    
    device, model, optimizer = modelsetup(learning_rate)

    # torch transforms
    ts = transform()

    # dataloaders
    training_dataloader = load_train_data(batch_size=batch_size, ts=ts)

    validation_dataloader = load_val_data(batch_size=batch_size, ts=ts)

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, labels in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    model.eval()
                    output = model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                val_loss /= len(validation_dataloader.dataset)
                val_acc = correct / len(validation_dataloader.dataset)

                print(
                    f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )

@main.command()
def infer():
    print("This is where the inference code will go")
