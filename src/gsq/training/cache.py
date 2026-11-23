import os

import hydra
import torch
from omegaconf import DictConfig

from gsq.model.encoders import PCAReservoirEncoder, ReservoirEncoder
from gsq.model.reservoir_cache import (
    derive_pca_reservoir_caches,
    load_or_compute_reservoir,
)
from gsq.utils.logs import get_logger
from gsq.utils.seeding import seed_if_configured

from .reservoir import (
    reservoir_cache_path,
    reservoir_cache_paths,
    source_reservoir_cache_path,
)

log = get_logger(__name__)


def _cache_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Cache device must be one of: auto, cpu, cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cache device is cuda, but CUDA is unavailable.")
    return torch.device(requested)


def cache_reservoir(cfg: DictConfig) -> None:
    """Precompute one immutable full-graph GESN representation."""
    log.info(
        "CUDA assignment: CUDA_VISIBLE_DEVICES=%s visible_device_count=%d",
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        torch.cuda.device_count(),
    )
    if seed_if_configured(cfg.get("seed")):
        log.info("Seeded Python, NumPy, and PyTorch with seed=%s", cfg.seed)
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        loader=cfg.loader,
        _recursive_=False,
    )
    datamodule.setup(stage="fit")
    data = datamodule.dataset[0]
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    if not isinstance(encoder, (ReservoirEncoder, PCAReservoirEncoder)):
        raise TypeError(
            "GESN pretraining requires the gesn or gesn_pca encoder."
        )
    device = _cache_device(str(cfg.get("device", "auto")))
    log.info(
        "GESN cache compute device=%s nodes=%d edges=%d",
        device,
        data.num_nodes,
        data.num_edges,
    )
    if isinstance(encoder, PCAReservoirEncoder):
        widths = tuple(int(value) for value in cfg.projection_channels)
        if not widths:
            raise ValueError("projection_channels cannot be empty for gesn_pca.")
        source_path = source_reservoir_cache_path(cfg)
        if source_path is None:
            raise RuntimeError("Could not construct the source reservoir cache path.")
        paths = reservoir_cache_paths(cfg)
        output_paths = dict(zip(widths, paths, strict=True))
        hits = derive_pca_reservoir_caches(
            encoder,
            data,
            source_path,
            output_paths,
            compute_device=device,
        )
        for width, path in output_paths.items():
            size_mib = path.stat().st_size / (1024 * 1024)
            log.info(
                "PCA GESN cache %s: source=%s path=%s shape=(%d, %d) "
                "size=%.1f MiB",
                "reused" if hits[width] else "created",
                source_path,
                path,
                data.num_nodes,
                width,
                size_mib,
            )
        return

    cache_path = reservoir_cache_path(cfg)
    if cache_path is None:
        raise RuntimeError("Could not construct a reservoir cache path.")

    embeddings, cache_hit = load_or_compute_reservoir(
        encoder,
        data,
        cache_path,
        compute_device=device,
    )
    size_mib = cache_path.stat().st_size / (1024 * 1024)
    log.info(
        "GESN cache %s: path=%s shape=%s dtype=%s device=%s size=%.1f MiB",
        "reused" if cache_hit else "created",
        cache_path,
        tuple(embeddings.shape),
        embeddings.dtype,
        embeddings.device,
        size_mib,
    )
