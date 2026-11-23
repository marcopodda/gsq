from time import perf_counter

import numpy as np
import torch
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier

from gsq.model.aggregators import KDEyMaximumLikelihood

from .structural import ppr_target_kernel_sum


class _GraphQuantifierBase(BaseQuantifier):
    def __init__(self) -> None:
        self.prediction_seconds = 0.0
        self.prediction_calls = 0
        self.predicted_bags = 0
        self.predicted_nodes = 0

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.predict_many([X])[0]

    def predict_many(self, samples):
        raise NotImplementedError

    def _prepare_samples(self, samples, num_nodes):
        samples = [torch.as_tensor(sample, dtype=torch.long) for sample in samples]
        if any(sample.ndim != 1 or sample.numel() == 0 for sample in samples):
            raise ValueError("Every sample must be a non-empty node-index vector.")
        if any((sample < 0).any() or (sample >= num_nodes).any() for sample in samples):
            raise ValueError(
                "A sample contains nodes outside the evaluation collection."
            )
        return samples

    def _record_prediction(self, started, samples) -> None:
        self.prediction_seconds += perf_counter() - started
        self.prediction_calls += 1
        self.predicted_bags += len(samples)
        self.predicted_nodes += sum(sample.numel() for sample in samples)

    def timing(self) -> dict[str, float | int]:
        seconds = self.prediction_seconds
        bags = self.predicted_bags
        nodes = self.predicted_nodes
        return {
            "prediction_seconds": seconds,
            "prediction_calls": self.prediction_calls,
            "predicted_bags": bags,
            "predicted_nodes": nodes,
            "seconds_per_bag": seconds / bags if bags else 0.0,
            "bags_per_second": bags / seconds if seconds else 0.0,
            "nodes_per_second": nodes / seconds if seconds else 0.0,
        }


class GraphQuantifier(_GraphQuantifierBase):
    def __init__(self, model, loader, mode):
        super().__init__()
        self.model = model.eval()
        self.loader = loader
        self.mode = mode

        embeddings, labels, edges = [], [], []
        full_edges, evaluation_indices = [], []
        full_node_offset = 0
        node_offset = 0

        with torch.inference_mode():
            for batch in loader:
                mask = batch[f"{mode}_mask"]
                if "n_id" in batch and mask.numel() != batch.x.size(0):
                    mask = mask[batch.n_id]
                embeddings.append(model(batch.x, batch.edge_index)[mask])
                labels.append(batch.y[mask])
                edges.append(_induced_edge_index(batch.edge_index, mask, node_offset))
                full_edges.append(batch.edge_index + full_node_offset)
                evaluation_indices.append(torch.where(mask)[0] + full_node_offset)
                node_offset += int(mask.sum())
                full_node_offset += batch.x.size(0)

        if not embeddings:
            raise ValueError("The loader produced no batches.")

        if len(embeddings) == 1:
            self.embeddings = embeddings[0].cpu()
            self.labels = labels[0].cpu()
        else:
            self.embeddings = torch.cat(embeddings, dim=0).cpu()
            self.labels = torch.cat(labels, dim=0).cpu()
        self.num_classes = getattr(model, "num_classes", int(self.labels.max()) + 1)
        self.edge_index = torch.cat(edges, dim=1).cpu().numpy()
        self.full_edge_index = torch.cat(full_edges, dim=1).cpu()
        self.evaluation_indices = torch.cat(evaluation_indices).cpu()
        self.num_nodes = full_node_offset

    @property
    def lc(self) -> LabelledCollection:
        instances = np.arange(self.labels.numel())
        labels = self.labels.numpy()
        return LabelledCollection(instances, labels, classes=range(self.num_classes))

    def predict_many(self, samples):
        """Predict several bags in one pooler/head pass."""
        started = perf_counter()
        samples = self._prepare_samples(samples, self.embeddings.size(0))
        if not samples:
            return np.empty((0, self.lc.n_classes), dtype=np.float32)

        with torch.inference_mode():
            sizes = torch.tensor(
                [sample.numel() for sample in samples], dtype=torch.long
            )
            batch = torch.arange(len(samples)).repeat_interleave(sizes)
            pooled = self.model.pooler(self.embeddings[torch.cat(samples)], batch)
            prevalence = self.model.predict_prevalence(pooled)
        result = prevalence.cpu().numpy()
        self._record_prediction(started, samples)
        return result


class AggregativeGraphQuantifier(_GraphQuantifierBase):
    def __init__(self, model, loader, mode, *, structural_protocol: bool = False):
        super().__init__()
        self.model = model.eval()
        self.structural_correction_enabled = bool(
            structural_protocol
            and isinstance(
                model.quantifier, KDEyMaximumLikelihood
            )
            and model.quantifier.structural_correction
        )
        probabilities, labels = [], []
        train_probabilities, train_labels = [], []
        calibration_probabilities, calibration_labels, edges = [], [], []
        full_edges, calibration_indices, evaluation_indices = [], [], []
        full_node_offset = 0
        node_offset = 0

        with torch.inference_mode():
            for batch in loader:
                node_probabilities = model.node_probabilities(batch)
                mode_mask = model._mask(batch, f"{mode}_mask")
                calibration_mask = model._mask(batch, "cal_mask")
                fit_mask = model._mask(batch, "train_mask") & ~calibration_mask
                probabilities.append(node_probabilities[mode_mask])
                labels.append(batch.y[mode_mask])
                edges.append(
                    _induced_edge_index(batch.edge_index, mode_mask, node_offset)
                )
                node_offset += int(mode_mask.sum())
                train_probabilities.append(node_probabilities[fit_mask])
                train_labels.append(batch.y[fit_mask])
                calibration_probabilities.append(node_probabilities[calibration_mask])
                calibration_labels.append(batch.y[calibration_mask])
                full_edges.append(batch.edge_index + full_node_offset)
                evaluation_indices.append(torch.where(mode_mask)[0] + full_node_offset)
                calibration_indices.append(
                    torch.where(calibration_mask)[0] + full_node_offset
                )
                full_node_offset += batch.x.size(0)

        if not probabilities:
            raise ValueError("The loader produced no batches.")
        self.probabilities = torch.cat(probabilities).cpu()
        self.labels = torch.cat(labels).cpu()
        self.edge_index = torch.cat(edges, dim=1).cpu().numpy()
        self.full_edge_index = torch.cat(full_edges, dim=1).cpu()
        self.num_nodes = full_node_offset
        self.evaluation_indices = torch.cat(evaluation_indices).cpu()
        self.calibration_indices = torch.cat(calibration_indices).cpu()
        self.calibration_probabilities = torch.cat(calibration_probabilities).cpu()
        self.calibration_labels = torch.cat(calibration_labels).cpu()
        model.quantifier.fit(
            torch.cat(train_probabilities),
            torch.cat(train_labels),
            torch.cat(calibration_probabilities),
            torch.cat(calibration_labels),
        )

    @property
    def lc(self) -> LabelledCollection:
        return LabelledCollection(
            np.arange(self.labels.numel()),
            self.labels.numpy(),
            classes=range(self.model.num_classes),
        )

    def predict_many(self, samples):
        started = perf_counter()
        samples = self._prepare_samples(samples, self.probabilities.size(0))
        if not samples:
            return np.empty((0, self.model.num_classes), dtype=np.float32)
        if self.structural_correction_enabled:
            with torch.inference_mode():
                result = self._predict_structurally_corrected(samples).numpy()
            self._record_prediction(started, samples)
            return result
        sizes = torch.tensor([sample.numel() for sample in samples], dtype=torch.long)
        group_index = torch.arange(len(samples)).repeat_interleave(sizes)
        with torch.inference_mode():
            prevalence = self.model.quantifier.predict_many(
                self.probabilities[torch.cat(samples)], group_index, len(samples)
            )
        result = prevalence.cpu().numpy()
        self._record_prediction(started, samples)
        return result

    def _predict_structurally_corrected(self, samples):
        quantifier = self.model.quantifier
        if not isinstance(quantifier, KDEyMaximumLikelihood):
            raise TypeError("Structural correction is only defined for KDEy.")

        estimates = []
        for sample in samples:
            full_sample = self.evaluation_indices[sample]
            kernel_sum = ppr_target_kernel_sum(
                self.full_edge_index,
                self.num_nodes,
                full_sample,
                steps=quantifier.correction_steps,
                restart_probability=quantifier.correction_restart_probability,
            ).to(self.calibration_probabilities.dtype)
            weights = kernel_sum[self.calibration_indices]
            weights = (
                quantifier.correction_kernel_weight * weights
                + 1
                - quantifier.correction_kernel_weight
            )
            estimates.append(
                quantifier.predict_with_local_weights(
                    self.probabilities[sample],
                    weights,
                )
            )
        return torch.stack(estimates)


def make_graph_quantifier(model, loader, mode, *, protocol_name: str | None = None):
    if hasattr(model, "quantifier") and hasattr(model, "classifier"):
        return AggregativeGraphQuantifier(
            model,
            loader,
            mode,
            structural_protocol=protocol_name in {"bfp", "rwp"},
        )
    return GraphQuantifier(model, loader, mode)


def _induced_edge_index(edge_index, mask, node_offset):
    """Relabel the mask-induced graph to collection-local node indices."""
    selected = torch.where(mask)[0]
    relabel = edge_index.new_full((mask.numel(),), -1)
    relabel[selected] = torch.arange(selected.numel(), device=edge_index.device)
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    return relabel[edge_index[:, keep]] + node_offset
