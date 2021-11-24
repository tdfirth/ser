from pathlib import Path

import torch
import typer
from torch import optim

from ser.models import Net, Parameters, TrainingModel, Data
from ser.train import train_model
from ser.tranforms import get_transforms
from utils import get_unique_id

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="Number of epochs to train over"
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch-size", help="Batch size to train with"
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning-rate", help="Learning rate to train with"
    ),
):
    parameters = Parameters(get_unique_id(), name, epochs, batch_size, learning_rate)

    print(f"Running experiment {name}")

    training_model = TrainingModel(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        _model=Net(),
        _optimizer=optim.Adam,
        parameters=parameters,
    )

    # torch transforms
    ts = get_transforms()

    # dataloaders
    data = Data.from_inputs(ts, parameters)

    train_model(parameters, data, training_model)



@main.command()
def infer():
    run_path = Path("./path/to/one/of/your/training/runs")
    label = 6

    # TODO load the parameters from the run_path so we can print them out!

    # select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")


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
