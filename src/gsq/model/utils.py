from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from quapy.data import LabelledCollection
from quapy.protocol import APP, UPP
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Batch
from torch_geometric.nn import MeanAggregation

from gsq.evaluation.protocols import (
    BFPProtocol,
    PreparedStructuralGraph,
    RWPProtocol,
    StructuralProtocol,
)


@dataclass
class SampledProtocolBags:
    indices: Tensor | list[Tensor]
    targets: Tensor


def _protocol_value(protocol: Mapping[str, Any] | Any, name: str) -> Any:
    return protocol[name] if isinstance(protocol, Mapping) else getattr(protocol, name)


def _protocol_stage_value(
    protocol: Mapping[str, Any] | Any,
    name: str,
    stage: str,
) -> Any:
    stage_name = f"{stage}_{name}"
    if isinstance(protocol, Mapping):
        return protocol.get(stage_name, protocol[name])
    return getattr(protocol, stage_name, getattr(protocol, name))


def _local_mask(batch: Batch, name: str, num_nodes: int) -> Tensor:
    mask = batch[name]
    if mask.numel() != num_nodes and "n_id" in batch:
        mask = mask[batch.n_id]  # ty: ignore[missing-attribute]
    if mask.numel() != num_nodes:
        raise ValueError(f"{name} and labels must refer to the same nodes.")
    return mask.bool()


def _stratified_mask_subsample(
    labels: Tensor,
    mask: Tensor,
    *,
    size: int,
    minimum_per_class: int,
    num_classes: int,
    random_state: int,
) -> Tensor:
    """Subsample a mask reproducibly while retaining its class proportions."""
    eligible = torch.where(mask)[0]
    if size >= eligible.numel():
        return mask
    if size < minimum_per_class * num_classes:
        raise ValueError(
            "The matched training split is too small for the requested seeds per class."
        )

    local_labels = labels[eligible].detach().cpu().numpy()
    counts = np.bincount(local_labels, minlength=num_classes)
    if counts.shape != (num_classes,) or np.any(counts < minimum_per_class):
        raise ValueError(
            "Every class needs at least seeds_per_class eligible training nodes."
        )

    ideal = size * counts / counts.sum()
    allocation = np.maximum(np.floor(ideal).astype(np.int64), minimum_per_class)
    while int(allocation.sum()) < size:
        deficit = ideal - allocation
        deficit[allocation >= counts] = -np.inf
        allocation[int(np.argmax(deficit))] += 1
    while int(allocation.sum()) > size:
        excess = allocation - ideal
        excess[allocation <= minimum_per_class] = -np.inf
        allocation[int(np.argmax(excess))] -= 1

    rng = np.random.default_rng(random_state)
    eligible_cpu = eligible.detach().cpu().numpy()
    selected = []
    for class_, class_size in enumerate(allocation):
        candidates = eligible_cpu[local_labels == class_].copy()
        rng.shuffle(candidates)
        selected.append(candidates[:class_size])
    selected_indices = torch.as_tensor(
        np.concatenate(selected), device=mask.device, dtype=torch.long
    )
    matched = torch.zeros_like(mask, dtype=torch.bool)
    matched[selected_indices] = True
    return matched


def _match_structural_training_mask(
    labels: Tensor,
    batch: Batch,
    mask: Tensor,
    *,
    protocol: Mapping[str, Any] | Any,
    num_classes: int,
    random_state: int,
) -> Tensor:
    """Match structural training-graph density to the validation split."""
    enabled = (
        protocol.get("match_train_split_size", False)
        if isinstance(protocol, Mapping)
        else getattr(protocol, "match_train_split_size", False)
    )
    if not bool(enabled):
        return mask
    validation_mask = _local_mask(batch, "val_mask", labels.numel())
    return _stratified_mask_subsample(
        labels,
        mask,
        size=int(validation_mask.sum()),
        minimum_per_class=(
            1
            if str(_protocol_value(protocol, "name")) == "rwp"
            else int(_protocol_stage_value(protocol, "seeds_per_class", "train"))
        ),
        num_classes=num_classes,
        random_state=random_state,
    )


def sample_protocol_indices(
    labels: Tensor,
    batch: Batch,
    *,
    mask: Tensor,
    num_classes: int,
    protocol: Mapping[str, Any] | Any,
    random_state: int,
    stage: str = "test",
    prepared_structural_graph: PreparedStructuralGraph | None = None,
) -> SampledProtocolBags:
    """Sample bags using the same protocol definitions as final evaluation."""
    name = str(_protocol_value(protocol, "name"))
    if mask.numel() != labels.numel() and "n_id" in batch:
        mask = mask[batch.n_id]  # ty: ignore[missing-attribute]
    if mask.numel() != labels.numel():
        raise ValueError("mask and labels must refer to the same nodes.")
    if name in {"bfp", "rwp"} and stage == "train":
        mask = _match_structural_training_mask(
            labels,
            batch,
            mask,
            protocol=protocol,
            num_classes=num_classes,
            random_state=random_state,
        )
    eligible = torch.where(mask)[0]
    if eligible.numel() == 0:
        raise ValueError("The selected split contains no nodes.")
    local_labels = labels[eligible].detach().cpu().numpy()
    collection = LabelledCollection(np.arange(eligible.numel()), local_labels)
    sample_size = int(_protocol_value(protocol, "sample_size"))
    if name == "app" and num_classes == 2:
        sampler = APP(
            collection,
            sample_size=sample_size,
            repeats=int(_protocol_stage_value(protocol, "repeats", stage)),
            smooth_limits_epsilon=0.001,
            random_state=random_state,
        )
    elif name == "app":
        sampler = UPP(
            collection,
            sample_size=sample_size,
            repeats=int(_protocol_stage_value(protocol, "repeats", stage)),
            random_state=random_state,
        )
    elif name in {"bfp", "rwp"}:
        if prepared_structural_graph is None:
            edge_index = batch["edge_index"].detach().cpu()
            prepared_structural_graph = StructuralProtocol.prepare_full_graph(
                labels.numel(), edge_index.numpy()
            )
        # Structural traversal uses the complete prepared graph and returns only
        # evaluation endpoints. The induced edge list was previously stored by
        # the protocol base class but never read during BFP or RWP sampling.
        local_edges = np.empty((2, 0), dtype=np.int64)
        common: dict[str, Any] = {
            "data": collection,
            "edge_index": local_edges,
            "sample_size": sample_size,
            "random_state": random_state,
            "prepared_full_graph": prepared_structural_graph,
        }
        if name == "bfp":
            sampler = BFPProtocol(
                seeds_per_class=int(
                    _protocol_stage_value(protocol, "seeds_per_class", stage)
                ),
                max_depth=int(_protocol_value(protocol, "max_depth")),
                evaluation_nodes=eligible.cpu().numpy(),
                **common,
            )
        else:
            sampler = RWPProtocol(
                seeds_per_class=None,
                num_seeds=int(_protocol_stage_value(protocol, "num_seeds", stage)),
                restart_probability=float(
                    _protocol_value(protocol, "restart_probability")
                ),
                max_steps=int(_protocol_value(protocol, "max_steps")),
                evaluation_nodes=eligible.cpu().numpy(),
                **common,
            )
    else:
        raise ValueError(f"Unsupported bag protocol: {name!r}.")

    sampled = list(sampler())
    if not sampled:
        raise ValueError(f"Protocol {name!r} produced no bags.")
    local_groups, prevalences = zip(*sampled, strict=True)
    groups = [
        eligible[torch.as_tensor(index, device=eligible.device)]
        for index in local_groups
    ]
    targets = torch.as_tensor(
        np.stack(prevalences), device=labels.device, dtype=torch.float32
    )
    if targets.shape != (len(groups), num_classes):
        raise RuntimeError(
            f"Protocol {name!r} returned targets with shape {tuple(targets.shape)}, "
            f"expected {(len(groups), num_classes)}."
        )
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise RuntimeError(f"Protocol {name!r} returned invalid prevalences.")
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if all(group.numel() == groups[0].numel() for group in groups):
        indices: Tensor | list[Tensor] = torch.stack(groups)
    else:
        indices = groups
    return SampledProtocolBags(indices=indices, targets=targets)


def pool_prevalence_indices(
    embeddings: Tensor,
    indices: Tensor | Sequence[Tensor],
    pooler: nn.Module,
    *,
    pooling_chunk_size: int = 32,
) -> Tensor:
    """Pool rectangular or variable-length node-index groups."""
    if pooling_chunk_size < 1:
        raise ValueError("pooling_chunk_size must be at least one.")
    rectangular = isinstance(indices, Tensor)
    groups_list: list[Tensor] = []
    if rectangular:
        if indices.ndim != 2 or indices.size(1) == 0:
            raise ValueError(
                "indices must contain non-empty groups with shape [groups, nodes]."
            )
        num_groups = indices.size(0)
    else:
        groups_list = list(indices)
        if not groups_list or any(
            group.ndim != 1 or group.numel() == 0 for group in groups_list
        ):
            raise ValueError("indices must contain non-empty one-dimensional groups.")
        num_groups = len(groups_list)
    pooled_chunks = []
    checkpoint_chunks = (
        bool(getattr(pooler, "checkpoint_chunks", False))
        and pooler.training
        and torch.is_grad_enabled()
    )
    effective_chunk_size = pooling_chunk_size
    max_nodes_per_chunk = getattr(pooler, "max_nodes_per_chunk", None)
    if rectangular and max_nodes_per_chunk is not None:
        group_size = indices.size(1)
        effective_chunk_size = min(
            pooling_chunk_size,
            max(1, int(max_nodes_per_chunk) // group_size),
        )
    for start in range(0, num_groups, effective_chunk_size):
        rectangular_indices = (
            indices[start : start + effective_chunk_size] if rectangular else None
        )
        if rectangular_indices is not None:
            selected = embeddings[rectangular_indices]
            if isinstance(pooler, MeanAggregation):
                pooled_chunks.append(selected.mean(dim=1))
                continue
            rectangular_forward = getattr(pooler, "forward_rectangular", None)
            if rectangular_forward is not None:
                pooled_chunks.append(
                    checkpoint(
                        rectangular_forward,
                        selected,
                        use_reentrant=False,
                    )
                    if checkpoint_chunks
                    else rectangular_forward(selected)
                )
                continue
            groups_in_chunk, group_size = rectangular_indices.shape
            group_index = torch.arange(
                groups_in_chunk, device=embeddings.device
            ).repeat_interleave(group_size)
            pooled_chunks.append(pooler(selected.flatten(0, 1), group_index))
            continue
        groups = groups_list[start : start + effective_chunk_size]
        sizes = torch.tensor(
            [group.numel() for group in groups], device=embeddings.device
        )
        selected = torch.cat(groups)
        group_index = torch.arange(
            len(groups), device=embeddings.device
        ).repeat_interleave(sizes)
        selected_embeddings = embeddings[selected]
        pooled_chunks.append(
            checkpoint(
                pooler,
                selected_embeddings,
                group_index,
                use_reentrant=False,
            )
            if checkpoint_chunks
            else pooler(selected_embeddings, group_index)
        )
    return torch.cat(pooled_chunks)
