from pathlib import Path
from typing import Any, cast

import hydra
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader, NeighborLoader

from gsq.training.reservoir import reservoir_cache_path


def load_experiment_data(
    root: Path,
) -> tuple[DataLoader | NeighborLoader, Any, DictConfig]:
    """Restore one explicitly requested training run on CPU."""
    config_path = root / "config.yaml"
    checkpoint_path = root / "model.ckpt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{root} must contain config.yaml and model.ckpt from a completed run."
        )
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping in {config_path}.")
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        loader=cfg.loader,
        _recursive_=False,
    )
    datamodule.setup(stage="test")
    loader = cast(DataLoader | NeighborLoader, datamodule.test_dataloader())

    model_class = cast(type[LightningModule], hydra.utils.get_class(cfg.model._target_))
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )
    cache_path = reservoir_cache_path(cfg)
    if cache_path is not None:
        cast(Any, model).prepare_reservoir(
            datamodule.dataset[0],
            str(cache_path),
            require_cached=True,
        )
    return loader, model.eval(), cfg
