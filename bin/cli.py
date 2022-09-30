import torch
from torch import optim
import json
from dataclasses import dataclass, asdict
from ser.model import Net
from ser.data import get_data
from ser.transforms import get_transforms
from ser.train import train_model
import git
import typer

@dataclass
class hparams_cls:
    epochs: int
    batch_size: int
    learning_rate: float

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

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

    # save hyperparameters and git hash
    hparams = hparams_cls(epochs, batch_size, learning_rate)
    hparams_dict = hparams.dict()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    hparams_dict['git hash'] = sha
    with open('./experiments/hyperparams_'+name+'.json', 'w') as f:
        json.dump(hparams_dict, f)

    # load model
    model = Net().to(device)

    # setup optimiser
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)

    # get torch transforms
    ts = get_transforms()

    # get dataloaders
    training_dataloader, validation_dataloader = get_data(hparams.batch_size, ts)

    # train model
    model = train_model(validation_dataloader, training_dataloader, model, optimizer, hparams.epochs, device, name)

    # save model
    torch.save(model.state_dict(), './experiments/model_'+name)

@main.command()
def infer():
    print("This is where the inference code will go")
