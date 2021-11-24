import json
import subprocess
from datetime import datetime
import uuid


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_unique_id() -> str:
    return (
        datetime.now().strftime("%Y-%m-%d_%H:%M")
        + "_"
        + get_git_revision_hash()
        + "_"
        + uuid.uuid4().hex  # TODO: possibly overkill?
    )


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
