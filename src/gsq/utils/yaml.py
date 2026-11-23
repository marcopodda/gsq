from pathlib import Path
from typing import Any

import yaml


def dump_yaml(file_path: Path, obj: dict[str, Any]) -> None:
    with file_path.open("w") as file:
        yaml.safe_dump(obj, file, sort_keys=False)
