from pathlib import Path
import torch
import os

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

from datetime import datetime


def savemodel(model,name):
    time = datetime.now()
    SAVE_DIR = EXPERIMENT_DIR / str(name) /  time.strftime("%Y-%m-%d") / time.strftime("%H:%M:%S")
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    torch.save(model.state_dict(), SAVE_DIR / "data.pt")
