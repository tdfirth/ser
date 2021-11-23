from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = "outputs"
DATA_DIR = PROJECT_ROOT / "data"
PARAMETER_DIR = PROJECT_ROOT / OUTPUTS_DIR / "parameters"
RESULTS_DIR = PROJECT_ROOT / OUTPUTS_DIR / "results"
Transform = Any
