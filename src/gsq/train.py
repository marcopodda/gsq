"""Command-line entry point for model training."""

import hydra
from omegaconf import DictConfig

import gsq


@hydra.main(
    version_base="1.3",
    config_path=gsq.CONFIG_DIR.resolve().as_posix(),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> None:
    from gsq.training.runner import train

    train(cfg)


if __name__ == "__main__":
    main()
