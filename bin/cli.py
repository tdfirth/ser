import typer
#import sys
#sys.path.append('../')
from ser.infer import inference

main = typer.Typer()


@main.command()
def train():
    print("This is where the training code will go")


@main.command()
def infer():
    inference()
    pass
