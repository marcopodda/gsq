"""Command-line entry point for reservoir precomputation."""

import json
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

import gsq
from gsq.training.cache import cache_reservoir
from gsq.training.reservoir import reservoir_cache_paths


def _write_completion_marker(cfg: DictConfig) -> None:
    output_dir = Path(cfg.paths.output_dir)
    marker = output_dir / "cache_complete.json"
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "task": "cache_reservoir",
                "status": "completed",
                "cache_paths": [str(path) for path in reservoir_cache_paths(cfg)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(marker)


@hydra.main(
    version_base="1.3",
    config_path=gsq.CONFIG_DIR.resolve().as_posix(),
    config_name="cache_reservoir.yaml",
)
def main(cfg: DictConfig) -> None:
    cache_reservoir(cfg)
    _write_completion_marker(cfg)


if __name__ == "__main__":
    main()
