import typer
from infer import print_text

main = typer.Typer()


@main.command()
def train():
    print("This is where the training code will go")


@main.command()
def infer():
    print_text()
