import json
import os
from pathlib import Path

from ser.models import Parameters


def get_params_in_dir(root_dir):
    for exp in os.listdir(root_dir):
        with open(root_dir / exp / "parameters.json") as f:
            parameters = Parameters(**json.load(f))
        print(parameters)
