import torch
from torch import optim
import json

from ser.model import Net
from ser.data import get_data
from ser.transforms import get_transforms
from ser.train import train_model
import typer

main = typer.Typer()

@main.command()
def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = get_transforms()

    # dataloaders
    training_dataloader, validation_dataloader = get_data(batch_size, ts)

    # train
    model = train_model(validation_dataloader, training_dataloader, model, optimizer, epochs, device, name)

    # save model
    torch.save(model.state_dict(), './experiments/model_'+name)

    # save hyperparameters
    hparams = {}
    for hparam in ["epochs", "batch_size", "learning_rate"]:
        hparams[hparam] = eval(hparam)

    with open('./experiments/hyperparams_'+name+'.json', 'w') as f:
        json.dump(hparams, f)

@main.command()
def infer():
    print("This is where the inference code will go")
