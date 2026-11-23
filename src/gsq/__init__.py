"""GSQ: reproducible graph symmetric quantification experiments."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parents[1]
_default_root = SOURCE_ROOT if (SOURCE_ROOT / "configs").is_dir() else Path.cwd()
PROJECT_ROOT = Path(os.environ.get("GSQ_ROOT", _default_root)).expanduser().resolve()
CONFIG_DIR = (
    PROJECT_ROOT / "configs"
    if (PROJECT_ROOT / "configs").is_dir()
    else PACKAGE_DIR / "configs"
)
SPLITS_DIR = (
    PROJECT_ROOT / "data" / "splits"
    if (PROJECT_ROOT / "data" / "splits").is_dir()
    else PACKAGE_DIR / "splits"
)
DATA_DIR = Path(
    os.environ.get("GSQ_DATA_DIR", Path.home() / ".cache" / "gsq" / "data")
).expanduser().resolve()
CACHE_DIR = Path(
    os.environ.get("GSQ_CACHE_DIR", Path.home() / ".cache" / "gsq")
).expanduser().resolve()
RUNS_DIR = Path(
    os.environ.get("GSQ_RUNS_DIR", PROJECT_ROOT / "runs")
).expanduser().resolve()

for name, value in {
    "PROJECT_ROOT": PROJECT_ROOT,
    "CONFIG_DIR": CONFIG_DIR,
    "DATA_DIR": DATA_DIR,
    "CACHE_DIR": CACHE_DIR,
    "LOGS_DIR": RUNS_DIR,
}.items():
    os.environ.setdefault(name, str(value))

__version__ = "0.1.0"
