from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops


def ppr_target_kernel_sum(
    edge_index: Tensor,
    num_nodes: int,
    targets: Tensor,
    *,
    steps: int,
    restart_probability: float,
) -> Tensor:
    """Sum finite-PPR kernel values from every target to every graph node.

    This is the grouped-weight SIS estimator from Damke and Hüllermeier: the
    target KDE is an unnormalized kernel sum here because subsequent weighted
    class-conditional averages cancel any common density normalizer.
    """
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("targets must be a non-empty node-index vector.")
    edges, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    source, target = edges
    degree = torch.bincount(source, minlength=num_nodes).clamp_min(1)
    restart = torch.zeros(num_nodes, dtype=torch.float64, device=edge_index.device)
    restart.index_add_(
        0,
        targets,
        torch.ones(targets.numel(), dtype=restart.dtype, device=restart.device),
    )
    mass = restart.clone()
    for _ in range(steps):
        propagated = torch.zeros_like(mass)
        propagated.index_add_(
            0, target, mass[source] / degree[source].to(mass.dtype)
        )
        mass = restart_probability * restart + (1 - restart_probability) * propagated
    return mass
