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
