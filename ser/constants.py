from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
PARAMETER_NAME = "parameters"
MODEL_NAME = "model"
Transform = Any
