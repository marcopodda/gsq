from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, MeanMetric, Metric, MinMetric

from gsq.evaluation.protocols import PreparedStructuralGraph, StructuralProtocol
from gsq.evaluation.structural import ppr_target_kernel_sum

from .aggregators import KDEyMaximumLikelihood
from .calibration import (
    CALIBRATION_MODES,
    PrevalenceCalibrator,
)
from .encoders import PCAReservoirEncoder, ReservoirEncoder
from .metrics import MeanRAE
from .reservoir_cache import load_or_compute_reservoir
from .utils import (
    SampledProtocolBags,
    pool_prevalence_indices,
    sample_protocol_indices,
)


def _regularize_pooled_output(
    pooled: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    return F.dropout(pooled, p=0.2, training=training)


def _make_calibrator(mode: str, num_classes: int) -> torch.nn.Module:
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"Unsupported calibration mode: {mode}.")
    if mode == "none":
        return torch.nn.Identity()
    return PrevalenceCalibrator(num_classes, classwise=mode == "classwise")


class QuantificationNetwork(LightningModule):
    def __init__(
        self,
        encoder: DictConfig,
        pooler: DictConfig,
        head: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        num_classes: int,
        pooling_chunk_size: int = 32,
        evaluation_seed: int = 0,
        calibration_mode: str = "none",
        calibration_max_steps: int = 100,
        calibration_regularization: float = 1e-3,
        bag_protocol: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.encoder = hydra.utils.instantiate(encoder)
        self.pooler = hydra.utils.instantiate(pooler)
        self.head = hydra.utils.instantiate(head)
        self.calibrator = _make_calibrator(calibration_mode, num_classes)
        self.calibration_mode = calibration_mode
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.num_classes = num_classes
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.bag_protocol = bag_protocol or DictConfig(
            {
                "name": "app",
                "sample_size": 500,
                "repeats": 100,
                "train_repeats": 128,
                "validation_repeats": 100,
            }
        )
        self.pooling_chunk_size = pooling_chunk_size
        self.evaluation_seed = evaluation_seed
        if calibration_max_steps < 1:
            raise ValueError("Calibration steps must be positive.")
        if calibration_regularization < 0:
            raise ValueError("Calibration regularization must be nonnegative.")
        self.calibration_max_steps = calibration_max_steps
        self.calibration_regularization = calibration_regularization

        self.train_metrics = self._make_metrics()
        self.val_metrics = self._make_metrics()
        self.test_metrics = self._make_metrics()
        self.val_ae_best = MinMetric()
        self.register_buffer("_reservoir_embeddings", None, persistent=False)
        self._protocol_sample_cache: dict[
            tuple[str, int, str], SampledProtocolBags
        ] = {}
        self._prepared_structural_graph: PreparedStructuralGraph | None = None
        self.save_hyperparameters(logger=False)

    @staticmethod
    def _make_metrics() -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {
                "loss": MeanMetric(),
                "ae": MeanAbsoluteError(),
                "rae": MeanRAE(),
            }
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self._reservoir_embeddings is not None:
            if x.size(0) != self._reservoir_embeddings.size(0):
                raise RuntimeError(
                    "Cached reservoir embeddings require the full graph. "
                    "Use _step with NeighborLoader batches so n_id can be applied."
                )
            return self._reservoir_embeddings
        return self.encoder(x, edge_index)

    def prepare_reservoir(
        self,
        data: Any,
        cache_path: str | None = None,
        *,
        require_cached: bool = False,
    ) -> bool | None:
        """Compute frozen full-graph reservoir states before optimization starts."""
        if not isinstance(self.encoder, (ReservoirEncoder, PCAReservoirEncoder)):
            return None
        self.encoder.eval()
        if cache_path is not None:
            self._reservoir_embeddings, cache_hit = load_or_compute_reservoir(
                self.encoder,
                data,
                Path(cache_path),
                require_cached=require_cached,
            )
            return cache_hit
        # These cached states are later consumed by trainable poolers/heads.
        # no_grad keeps them detached without turning them into inference tensors,
        # which autograd cannot save as inputs for parameter gradients.
        with torch.no_grad():
            self._reservoir_embeddings = self.encoder(data.x, data.edge_index).detach()
        return False

    def _encode_batch(self, batch: Any) -> torch.Tensor:
        if self._reservoir_embeddings is None:
            return self(batch.x, batch.edge_index)
        if "n_id" in batch:
            return self._reservoir_embeddings[batch.n_id]
        return self(batch.x, batch.edge_index)

    def _structural_graph(self, batch: Any) -> PreparedStructuralGraph | None:
        protocol_name = str(self.bag_protocol.get("name"))
        if protocol_name not in {"bfp", "rwp"}:
            return None
        if self._prepared_structural_graph is None:
            edge_index = batch.edge_index.detach().cpu().numpy()
            self._prepared_structural_graph = StructuralProtocol.prepare_full_graph(
                batch.y.numel(), edge_index
            )
        return self._prepared_structural_graph

    def _step(
        self,
        batch: Any,
        split: str,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self._encode_batch(batch)
        mask = batch[f"{split}_mask"]
        if split == "train":
            mask = mask & ~batch["cal_mask"]
        protocol_seed = (
            self.evaluation_seed + self.current_epoch
            if split == "train"
            else self.evaluation_seed + batch_idx
        )
        cache_key = (split, batch_idx, str(batch.y.device))
        sampled = self._protocol_sample_cache.get(cache_key)
        if sampled is None or split == "train":
            sampled = sample_protocol_indices(
                batch.y,
                batch,
                mask=mask,
                num_classes=self.num_classes,
                protocol=self.bag_protocol,
                random_state=protocol_seed,
                stage="train" if split == "train" else "validation",
                prepared_structural_graph=self._structural_graph(batch),
            )
            if split != "train":
                self._protocol_sample_cache[cache_key] = sampled
        pooled = self._pool(embeddings, sampled.indices)
        target = sampled.targets.to(embeddings.dtype)
        loss, raw_prediction = self.head(pooled, target)
        prediction = self._calibrate_prediction(raw_prediction)
        return loss, prediction.detach(), target.detach()

    def predict_prevalence(self, pooled: torch.Tensor) -> torch.Tensor:
        return self._calibrate_prediction(self.head.predict(pooled))

    def _calibrate_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.calibrator(prediction)

    @property
    def selected_calibration(self) -> str:
        return self.calibration_mode

    def fit_calibrators(self, data: Any) -> dict[str, float]:
        """Fit the explicitly configured bag-level scaler."""
        prediction, target = self._predict_protocol_bags(
            data,
            "cal_mask",
            seed=self.evaluation_seed,
        )
        if isinstance(self.calibrator, PrevalenceCalibrator):
            self.calibrator.fit(
                prediction,
                target,
                max_steps=self.calibration_max_steps,
                regularization=self.calibration_regularization,
            )
        calibrated = self.calibrator(prediction)
        val_prediction, val_target = self._predict_protocol_bags(
            data, "val_mask", seed=self.evaluation_seed
        )
        return {
            "ae/cal": F.l1_loss(calibrated, target).item(),
            "ae/val": F.l1_loss(self.calibrator(val_prediction), val_target).item(),
        }

    def _pool(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        pooled = pool_prevalence_indices(
            embeddings,
            indices,
            self.pooler,
            pooling_chunk_size=self.pooling_chunk_size,
        )
        return _regularize_pooled_output(pooled, training=self.training)

    def _predict_protocol_bags(
        self,
        data: Any,
        mask_name: str,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            embeddings = self(data.x, data.edge_index)
            sampled = sample_protocol_indices(
                data.y,
                data,
                mask=data[mask_name],
                num_classes=self.num_classes,
                protocol=self.bag_protocol,
                random_state=seed,
                stage="validation",
                prepared_structural_graph=self._structural_graph(data),
            )
            pooled = self._pool(embeddings, sampled.indices)
            prediction = self.head.predict(pooled)
        return prediction, sampled.targets.to(prediction.dtype)

    def _update_metrics(self, metrics, loss, prediction, target) -> None:
        metrics["loss"].update(loss.detach(), weight=prediction.size(0))
        metrics["ae"].update(prediction, target)
        metrics["rae"].update(prediction, target)

    def _log_metrics(self, metrics: torch.nn.ModuleDict, split: str) -> None:
        for name, metric in metrics.items():
            metric = cast(Metric, metric)
            self.log(
                f"{name}/{split}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name != "loss",
                logger=True,
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, prediction, target = self._step(batch, "train", batch_idx)
        self._update_metrics(self.train_metrics, loss, prediction, target)
        self._log_metrics(self.train_metrics, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, prediction, target = self._step(batch, "val", batch_idx)
        self._update_metrics(self.val_metrics, loss, prediction, target)
        self._log_metrics(self.val_metrics, "val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        val_ae = cast(Metric, self.val_metrics["ae"]).compute()
        self.val_ae_best.update(val_ae)
        best = cast(Metric, self.val_ae_best).compute()
        self.log("ae/val_best", best, prog_bar=True, logger=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, prediction, target = self._step(batch, "test", batch_idx)
        self._update_metrics(self.test_metrics, loss, prediction, target)
        self._log_metrics(self.test_metrics, "test")

    def configure_optimizers(self) -> Any:
        optimizer = hydra.utils.instantiate(
            self.optimizer_config,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = self.scheduler_config.get("scheduler")
        if not scheduler:
            return {"optimizer": optimizer}
        scheduler = hydra.utils.instantiate(
            scheduler,
            optimizer=optimizer,
            _convert_="partial",
        )
        config = {"scheduler": scheduler}
        config.update(self.scheduler_config.get("extras", {}))
        return {"optimizer": optimizer, "lr_scheduler": config}

    def on_fit_end(self) -> None:
        # The cache can dominate memory on large graphs and is reproducible from
        # the frozen reservoir parameters stored in the checkpoint.
        self._reservoir_embeddings = None
        self._protocol_sample_cache.clear()


class AggregativeQuantificationNetwork(LightningModule):
    """Train a node classifier and quantify bags through an aggregation rule."""

    def __init__(
        self,
        encoder: DictConfig,
        classifier: DictConfig,
        quantifier: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        num_classes: int,
        evaluation_seed: int = 0,
        calibration_mode: str = "none",
        calibration_max_steps: int = 100,
        calibration_regularization: float = 1e-3,
        bag_protocol: DictConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.encoder = hydra.utils.instantiate(encoder)
        self.classifier = hydra.utils.instantiate(classifier)
        self.quantifier = hydra.utils.instantiate(quantifier)
        self.calibrator = _make_calibrator(calibration_mode, num_classes)
        self.calibration_mode = calibration_mode
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.num_classes = num_classes
        self.bag_protocol = bag_protocol or DictConfig(
            {
                "name": "app",
                "sample_size": 500,
                "repeats": 100,
                "train_repeats": 128,
                "validation_repeats": 100,
            }
        )
        self.evaluation_seed = evaluation_seed
        if calibration_max_steps < 1:
            raise ValueError("Calibration steps must be positive.")
        if calibration_regularization < 0:
            raise ValueError("Calibration regularization must be nonnegative.")
        self.calibration_max_steps = calibration_max_steps
        self.calibration_regularization = calibration_regularization
        self.val_metrics = self._make_metrics()
        self.test_metrics = self._make_metrics()
        self.val_ae_best = MinMetric()
        self.register_buffer("_reservoir_embeddings", None, persistent=False)
        self._protocol_sample_cache: dict[
            tuple[str, int, str], SampledProtocolBags
        ] = {}
        self._prepared_structural_graph: PreparedStructuralGraph | None = None
        self.save_hyperparameters(logger=False)

    @staticmethod
    def _make_metrics() -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(
            {"loss": MeanMetric(), "ae": MeanAbsoluteError(), "rae": MeanRAE()}
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self._reservoir_embeddings is not None:
            if x.size(0) != self._reservoir_embeddings.size(0):
                raise RuntimeError(
                    "Cached reservoir embeddings require the full graph."
                )
            return self._reservoir_embeddings
        return self.encoder(x, edge_index)

    def prepare_reservoir(
        self,
        data: Any,
        cache_path: str | None = None,
        *,
        require_cached: bool = False,
    ) -> bool | None:
        if not isinstance(self.encoder, (ReservoirEncoder, PCAReservoirEncoder)):
            return None
        self.encoder.eval()
        if cache_path is not None:
            self._reservoir_embeddings, cache_hit = load_or_compute_reservoir(
                self.encoder,
                data,
                Path(cache_path),
                require_cached=require_cached,
            )
            return cache_hit
        with torch.inference_mode():
            self._reservoir_embeddings = self.encoder(data.x, data.edge_index).detach()
        return False

    def _encode_batch(self, batch: Any) -> torch.Tensor:
        if self._reservoir_embeddings is None:
            return self(batch.x, batch.edge_index)
        if "n_id" in batch:
            return self._reservoir_embeddings[batch.n_id]
        return self(batch.x, batch.edge_index)

    def _structural_graph(self, batch: Any) -> PreparedStructuralGraph | None:
        protocol_name = str(self.bag_protocol.get("name"))
        if protocol_name not in {"bfp", "rwp"}:
            return None
        if self._prepared_structural_graph is None:
            edge_index = batch.edge_index.detach().cpu().numpy()
            self._prepared_structural_graph = StructuralProtocol.prepare_full_graph(
                batch.y.numel(), edge_index
            )
        return self._prepared_structural_graph

    @staticmethod
    def _mask(batch: Any, name: str) -> torch.Tensor:
        mask = batch[name]
        if mask.numel() != batch.y.numel() and "n_id" in batch:
            mask = mask[batch.n_id]
        if mask.numel() != batch.y.numel():
            raise ValueError(f"{name} and labels must refer to the same nodes.")
        return mask

    def node_logits(self, batch: Any) -> torch.Tensor:
        return self.classifier(self._encode_batch(batch))

    def node_probabilities(self, batch: Any) -> torch.Tensor:
        probabilities = self.node_logits(batch).softmax(dim=-1).float()
        return self._calibrate_node_probabilities(probabilities)

    def _calibrate_node_probabilities(
        self, probabilities: torch.Tensor
    ) -> torch.Tensor:
        return self.calibrator(probabilities)

    @property
    def selected_calibration(self) -> str:
        return self.calibration_mode

    @staticmethod
    def _grouped_indices(
        indices: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        groups = (
            list(indices.unbind()) if isinstance(indices, torch.Tensor) else list(indices)
        )
        sizes = torch.tensor(
            [group.numel() for group in groups], device=groups[0].device
        )
        group_index = torch.arange(
            len(groups), device=groups[0].device
        ).repeat_interleave(sizes)
        return groups, group_index

    def _protocol_predictions(
        self,
        probabilities: torch.Tensor,
        batch: Any,
        mask_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled = sample_protocol_indices(
            batch.y,
            batch,
            mask=self._mask(batch, mask_name),
            num_classes=self.num_classes,
            protocol=self.bag_protocol,
            random_state=self.evaluation_seed,
            stage="validation",
            prepared_structural_graph=self._structural_graph(batch),
        )
        groups, group_index = self._grouped_indices(sampled.indices)
        if (
            isinstance(self.quantifier, KDEyMaximumLikelihood)
            and self.quantifier.structural_correction
            and str(self.bag_protocol.get("name")) in {"bfp", "rwp"}
        ):
            calibration_indices = torch.where(self._mask(batch, "cal_mask"))[0]
            prediction = torch.stack(
                [
                    self._predict_kdey_sis(
                        probabilities[group], group, calibration_indices, batch
                    )
                    for group in groups
                ]
            )
        else:
            prediction = self.quantifier.predict_many(
                probabilities[torch.cat(groups)], group_index, len(groups)
            )
        return prediction, sampled.targets.to(prediction.dtype)

    def _predict_kdey_sis(
        self,
        sample_probabilities: torch.Tensor,
        sample_indices: torch.Tensor,
        calibration_indices: torch.Tensor,
        batch: Any,
    ) -> torch.Tensor:
        quantifier = self.quantifier
        if not isinstance(quantifier, KDEyMaximumLikelihood):
            raise TypeError("SIS prediction requires KDEyMaximumLikelihood.")
        kernel_sum = ppr_target_kernel_sum(
            batch.edge_index,
            batch.y.numel(),
            sample_indices,
            steps=quantifier.correction_steps,
            restart_probability=quantifier.correction_restart_probability,
        ).to(sample_probabilities.dtype)
        weights = kernel_sum[calibration_indices]
        weights = (
            quantifier.correction_kernel_weight * weights
            + 1
            - quantifier.correction_kernel_weight
        )
        return quantifier.predict_with_local_weights(sample_probabilities, weights)

    def fit_calibrators(self, data: Any) -> dict[str, float]:
        """Fit the configured node-probability scaler and aggregation rule."""
        self.eval()
        with torch.no_grad():
            raw_probabilities = self.node_logits(data).softmax(dim=-1).float()
        calibration_mask = self._mask(data, "cal_mask")
        fit_mask = self._mask(data, "train_mask") & ~calibration_mask
        calibration_target = F.one_hot(
            data.y[calibration_mask], num_classes=self.num_classes
        ).float()
        if isinstance(self.calibrator, PrevalenceCalibrator):
            self.calibrator.fit(
                raw_probabilities[calibration_mask],
                calibration_target,
                max_steps=self.calibration_max_steps,
                regularization=self.calibration_regularization,
            )
        probabilities = self.calibrator(raw_probabilities)
        self.quantifier.fit(
            probabilities[fit_mask],
            data.y[fit_mask],
            probabilities[calibration_mask],
            data.y[calibration_mask],
        )
        cal_prediction, cal_target = self._protocol_predictions(
            probabilities, data, "cal_mask"
        )
        val_prediction, val_target = self._protocol_predictions(
            probabilities, data, "val_mask"
        )
        return {
            "ae/cal": F.l1_loss(cal_prediction, cal_target).item(),
            "ae/val": F.l1_loss(val_prediction, val_target).item(),
        }

    def fit_quantifier(self, probabilities: torch.Tensor, batch: Any) -> None:
        calibration_mask = self._mask(batch, "cal_mask")
        fit_mask = self._mask(batch, "train_mask") & ~calibration_mask
        self.quantifier.fit(
            probabilities[fit_mask],
            batch.y[fit_mask],
            probabilities[calibration_mask],
            batch.y[calibration_mask],
        )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        logits = self.node_logits(batch)
        fit_mask = self._mask(batch, "train_mask") & ~self._mask(batch, "cal_mask")
        if not fit_mask.any():
            raise ValueError(
                "The classifier training split is empty after calibration holdout."
            )
        loss = F.cross_entropy(logits[fit_mask], batch.y[fit_mask])
        accuracy = (logits[fit_mask].argmax(dim=1) == batch.y[fit_mask]).float().mean()
        batch_size = int(fit_mask.sum())
        self.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "accuracy/train",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def _quantification_step(
        self, batch: Any, split: str, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.node_logits(batch)
        probabilities = self._calibrate_node_probabilities(
            logits.softmax(dim=-1).float()
        )
        self.fit_quantifier(probabilities, batch)
        split_mask = self._mask(batch, f"{split}_mask")
        cache_key = (split, batch_idx, str(batch.y.device))
        sampled = self._protocol_sample_cache.get(cache_key)
        if sampled is None:
            sampled = sample_protocol_indices(
                batch.y,
                batch,
                mask=split_mask,
                num_classes=self.num_classes,
                protocol=self.bag_protocol,
                random_state=self.evaluation_seed + batch_idx,
                stage="validation" if split == "val" else "test",
                prepared_structural_graph=self._structural_graph(batch),
            )
            self._protocol_sample_cache[cache_key] = sampled
        groups, group_index = self._grouped_indices(sampled.indices)
        prediction = self.quantifier.predict_many(
            probabilities[torch.cat(groups)], group_index, len(groups)
        )
        loss = F.cross_entropy(logits[split_mask], batch.y[split_mask])
        return loss, prediction, sampled.targets.to(prediction.dtype)

    @staticmethod
    def _update_metrics(metrics, loss, prediction, target) -> None:
        metrics["loss"].update(loss.detach(), weight=prediction.size(0))
        metrics["ae"].update(prediction, target)
        metrics["rae"].update(prediction, target)

    def _log_metrics(self, metrics: torch.nn.ModuleDict, split: str) -> None:
        for name, metric in metrics.items():
            self.log(
                f"{name}/{split}",
                cast(Metric, metric),
                on_step=False,
                on_epoch=True,
                prog_bar=name != "loss",
                logger=True,
            )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, prediction, target = self._quantification_step(batch, "val", batch_idx)
        self._update_metrics(self.val_metrics, loss, prediction, target)
        self._log_metrics(self.val_metrics, "val")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return
        val_ae = cast(Metric, self.val_metrics["ae"]).compute()
        self.val_ae_best.update(val_ae)
        best = cast(Metric, self.val_ae_best).compute()
        self.log("ae/val_best", best, prog_bar=True, logger=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        loss, prediction, target = self._quantification_step(batch, "test", batch_idx)
        self._update_metrics(self.test_metrics, loss, prediction, target)
        self._log_metrics(self.test_metrics, "test")

    def configure_optimizers(self) -> Any:
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters(), _convert_="partial"
        )
        scheduler = self.scheduler_config.get("scheduler")
        if not scheduler:
            return {"optimizer": optimizer}
        scheduler = hydra.utils.instantiate(
            scheduler, optimizer=optimizer, _convert_="partial"
        )
        config = {"scheduler": scheduler}
        config.update(self.scheduler_config.get("extras", {}))
        return {"optimizer": optimizer, "lr_scheduler": config}

    def on_fit_end(self) -> None:
        self._reservoir_embeddings = None
        self._protocol_sample_cache.clear()
