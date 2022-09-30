from pathlib import Path
from torchvision import  transforms

def transform():
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return(ts)
