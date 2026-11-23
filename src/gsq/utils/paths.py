from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def get_output_dir() -> Path:
    cfg = HydraConfig.get()
    output_dir = Path(cfg.runtime.output_dir)
    return output_dir
