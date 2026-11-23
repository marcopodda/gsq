import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _cache_path(cfg: DictConfig, encoder: dict[Any, Any]) -> Path:
    dataset = OmegaConf.to_container(cfg.datamodule.dataset, resolve=True)
    if not isinstance(dataset, dict):
        raise TypeError("Expected datamodule.dataset to resolve to a mapping.")
    dataset.pop("fold_index", None)
    key = json.dumps(
        {"schema": 1, "dataset": dataset, "encoder": encoder},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(key).hexdigest()[:20]
    return Path(cfg.paths.cache_dir) / "gesn" / f"{cfg.name}-{digest}.pt"


def reservoir_cache_path(cfg: DictConfig) -> Path | None:
    """Return the fold- and downstream-independent reservoir cache path."""
    encoder = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    if not isinstance(encoder, dict) or not str(encoder.get("_target_", "")).endswith(
        (".ReservoirEncoder", ".PCAReservoirEncoder")
    ):
        return None
    return _cache_path(cfg, encoder)


def source_reservoir_cache_path(cfg: DictConfig) -> Path | None:
    """Return the raw reservoir cache underlying a PCA-reservoir config."""
    encoder = OmegaConf.to_container(cfg.model.encoder, resolve=True)
    if not isinstance(encoder, dict) or not str(encoder.get("_target_", "")).endswith(
        ".PCAReservoirEncoder"
    ):
        return None
    source = {
        key: value
        for key, value in encoder.items()
        if key
        not in {
            "reservoir_channels",
            "pca_oversampling",
            "pca_iterations",
        }
    }
    source["_target_"] = "gsq.model.ReservoirEncoder"
    source["hidden_channels"] = encoder["reservoir_channels"]
    return _cache_path(cfg, source)


def reservoir_cache_paths(cfg: DictConfig) -> tuple[Path, ...]:
    """Return every cache artifact produced by one cache-reservoir job."""
    target = str(cfg.model.encoder.get("_target_", ""))
    projection_channels = cfg.get("projection_channels", [])
    if target.endswith(".PCAReservoirEncoder") and projection_channels:
        paths = []
        for width in projection_channels:
            projected_cfg = copy.deepcopy(cfg)
            projected_cfg.model.encoder.hidden_channels = int(width)
            path = reservoir_cache_path(projected_cfg)
            if path is None:
                raise RuntimeError("Could not construct a PCA reservoir cache path.")
            paths.append(path)
        return tuple(paths)
    path = reservoir_cache_path(cfg)
    return () if path is None else (path,)
