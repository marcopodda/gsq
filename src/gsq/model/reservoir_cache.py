from pathlib import Path
from typing import Any

import torch

from .encoders import PCAReservoirEncoder, ReservoirEncoder


def _save(
    encoder: ReservoirEncoder | PCAReservoirEncoder,
    embeddings: torch.Tensor,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state": {
                name: value.detach().cpu()
                for name, value in encoder.state_dict().items()
            },
            "embeddings": embeddings.detach().cpu(),
            "metadata": metadata or {},
        },
        path,
    )


def load_or_compute_reservoir(
    encoder: ReservoirEncoder | PCAReservoirEncoder,
    data: Any,
    cache_path: Path,
    *,
    require_cached: bool = False,
    compute_device: torch.device | None = None,
) -> tuple[torch.Tensor, bool]:
    """Load a GESN representation or compute it when allowed."""
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        embeddings = payload["embeddings"]
        if embeddings.shape != (data.num_nodes, encoder.hidden_channels):
            raise ValueError(f"Invalid GESN cache shape in {cache_path}.")
        encoder.load_state_dict(payload["encoder_state"], strict=True)
        return embeddings.to(data.x.device), True

    if require_cached or isinstance(encoder, PCAReservoirEncoder):
        raise FileNotFoundError(
            f"Missing GESN cache {cache_path}. Run `gsq pretrain-gesn` first."
        )

    device = compute_device or data.x.device
    graph = data.to(device)
    encoder.initialize(graph.edge_index)
    encoder.to(device).eval()
    with torch.inference_mode():
        embeddings = encoder(graph.x, graph.edge_index).detach()
    _save(encoder, embeddings, cache_path)
    return embeddings, False


def derive_pca_reservoir_caches(
    encoder: PCAReservoirEncoder,
    data: Any,
    source_cache_path: Path,
    output_paths: dict[int, Path],
    *,
    compute_device: torch.device,
) -> dict[int, bool]:
    """Project one cached GESN representation to the requested PCA widths."""
    hits = {width: path.exists() for width, path in output_paths.items()}
    missing = [width for width, hit in hits.items() if not hit]
    if not missing:
        return hits

    source_encoder = ReservoirEncoder(
        encoder.in_channels,
        encoder.reservoir_channels,
        encoder.recurrent_scale,
        encoder.input_scale,
        encoder.num_layers,
        encoder.max_iterations,
        encoder.random_seed,
    )
    raw, _ = load_or_compute_reservoir(
        source_encoder, data, source_cache_path, require_cached=True
    )
    raw = raw.to(compute_device, dtype=torch.float32)
    centered = raw - raw.mean(dim=0)
    maximum = max(output_paths)
    q = min(maximum + encoder.pca_oversampling, *centered.shape)
    devices = [compute_device.index or 0] if compute_device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(encoder.random_seed)
        _, _, components = torch.pca_lowrank(
            centered, q=q, center=False, niter=encoder.pca_iterations
        )
    components = components[:, :maximum]
    rows = components.abs().argmax(dim=0)
    signs = components[rows, torch.arange(maximum, device=components.device)].sign()
    projected = centered @ (components * signs)

    for width in missing:
        projected_encoder = PCAReservoirEncoder(
            encoder.in_channels,
            encoder.reservoir_channels,
            width,
            encoder.recurrent_scale,
            encoder.input_scale,
            encoder.num_layers,
            encoder.max_iterations,
            encoder.random_seed,
            encoder.pca_oversampling,
            encoder.pca_iterations,
        )
        _save(
            projected_encoder,
            projected[:, :width],
            output_paths[width],
            {"source_cache": str(source_cache_path), "output_channels": width},
        )
        hits[width] = False
    return hits
