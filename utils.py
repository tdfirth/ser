import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_unique_id() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H:%M')}_{get_git_revision_hash()}"


def write_dataclass_dict(class_object: Any, file_path: str):
    with open(file_path, "w") as f:
        f.write(json.dumps(asdict(class_object)))


def load_object_from_json(class_object: Any, file_path: Path):
    with open(file_path) as f:
        return class_object(**json.load(f))


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
