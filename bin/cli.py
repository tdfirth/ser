from pathlib import Path

from ser.train import train

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#or
#we could have structure
#def set params
#def run or train
def save_setup(params):
    print(params)
    #save shit to a .txt file so can reload the model
    return

@main.command() #to run: ser model-setup --name ect
def model_setup(name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."),
        epochs: int = typer.Option(
            2, "-e", "--epochs", help="number of epochs to train the model for."
        ),
        batch_size: int = typer.Option(
            1000, "-b", "--batch_size", help="batch size."
        ),
        learning_rate: float = typer.Option(
            0.01, "-lr", "--learning_rate", help="learning rate."
        ),
        DATA_DIR = DATA_DIR
        ):

    params = {"name":name, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate}
    print("\nData directory: ", DATA_DIR, "\n")
    
    train(name, epochs, batch_size, learning_rate, DATA_DIR)
    save_setup(params)

    return params

@main.command()
def inference():
    infer()
    pass
    