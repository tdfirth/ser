from datetime import datetime
from pathlib import Path

import typer
import torch
import git

from ser.train import train as run_train
from ser.constants import RESULTS_DIR
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize

from inference.image_select import image_select
from inference.inference import inference

import json

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
    run_path: Path = typer.Option(
        ..., "-p", "--path", help="File path to inference model."
    )
):
    # this command is now specified by a typer argument
    # run_path = Path("./path/to/one/of/your/training/runs")
    label = 6

    # TODO load the parameters from the run_path so we can print them out!
    parameters = json.load(open(run_path / "params.json"))
    print(parameters)

    # select image to run inference for- moved to image_select.py
    # dataloader = test_dataloader(1, transforms(normalize))
    # images, labels = next(iter(dataloader))
    # while labels[0].item() != label:
    #     images, labels = next(iter(dataloader))
    images, labels = image_select(label)

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference- moved to inference.py
    # model.eval()
    # output = model(images)
    # pred = output.argmax(dim=1, keepdim=True)[0].item()
    # certainty = max(list(torch.exp(output)[0]))
    # pixels = images[0][0]
    pred, certainty, pixels = inference(model,images)
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}, with certainty {certainty}.")
    


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
