"""Command-line entry point for model evaluation."""

import hydra
from omegaconf import DictConfig

import gsq
from gsq.evaluation.runner import evaluate


@hydra.main(
    version_base="1.3",
    config_path=gsq.CONFIG_DIR.resolve().as_posix(),
    config_name="eval.yaml",
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
