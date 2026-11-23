import hydra
from lightning.pytorch import Callback
from omegaconf import DictConfig


def instantiate_callbacks(config: DictConfig) -> list[Callback]:
    return [
        hydra.utils.instantiate(item)
        for item in config.values()
        if isinstance(item, DictConfig) and "_target_" in item
    ]
