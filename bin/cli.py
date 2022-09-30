from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

# importing model
from ser.model import Net
# importing transforms
from ser.transforms import ts
# importing data loaders
from ser.data import training_dataloader, validation_dataloader
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

    # train
    # for epoch in range(epochs):
    #     for i, (images, labels) in enumerate(training_dataloader):
    #         images, labels = images.to(device), labels.to(device)
    #         model.train()
    #         optimizer.zero_grad()
    #         output = model(images)
    #         loss = F.nll_loss(output, labels)
    #         loss.backward()
    #         optimizer.step()
    #         print(
    #             f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
    #             f"| Loss: {loss.item():.4f}"
    #         )
    #         # validate
    #         val_loss = 0
    #         correct = 0
    #         with torch.no_grad():
    #             for images, labels in validation_dataloader:
    #                 images, labels = images.to(device), labels.to(device)
    #                 model.eval()
    #                 output = model(images)
    #                 val_loss += F.nll_loss(output, labels, reduction="sum").item()
    #                 pred = output.argmax(dim=1, keepdim=True)
    #                 correct += pred.eq(labels.view_as(pred)).sum().item()
    #             val_loss /= len(validation_dataloader.dataset)
    #             val_acc = correct / len(validation_dataloader.dataset)

    #             print(
    #                 f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
    #             )
    trainer()

@main.command()
def infer():
    print("This is where the inference code will go")
