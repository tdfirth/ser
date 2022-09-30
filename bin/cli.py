import torch
import typer

from ser.model import model_setup
from ser.transforms import torch_transform
from ser.data import load_data
from ser.train import model_train

main = typer.Typer()

@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs to train."
    ),
    batch_size: int = typer.Option(
        1000, "-bs", "--batch-size", help="Number of samples in each training batch."
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning-rate", help="Step size at each iteration."
    ),

):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!


    # pull in model and optimizer
    model, optimizer = model_setup(device, learning_rate)

    # torch transforms
    ts = torch_transform()
    
    # dataloaders
    training_dataloader, validation_dataloader = load_data(ts, batch_size)

    # train
    model_train(epochs, training_dataloader, validation_dataloader, \
                device, model, optimizer)

@main.command()
def infer():
    print("This is where the inference code will go")
