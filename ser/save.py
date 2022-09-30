from pathlib import Path
import torch
import os
import json

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

from datetime import datetime

def foldersetup(name):
    time = datetime.now()
    SAVE_DIR = EXPERIMENT_DIR / str(name) /  time.strftime("%Y-%m-%d") / time.strftime("%H:%M:%S")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    return(SAVE_DIR)

def saveparams(name, epochs, batch_size, learning_rate):
    SAVE_DIR = foldersetup(name)
    params = {'epochs' : epochs,
              'batch_size' : batch_size,
              'learning_rate': learning_rate
    }
    with open(SAVE_DIR / "params.json", 'w') as f:
        json.dump(params, f)



def savemodel(model,name):
    SAVE_DIR = foldersetup(name)
    torch.save(model.state_dict(), SAVE_DIR / "data.pt")
