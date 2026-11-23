from abc import ABC, abstractmethod

import numpy as np
import quapy.functional as qf
import torch
from scipy.special import logsumexp
from torch import Tensor, nn


def _validate_probabilities(probabilities: Tensor, num_classes: int) -> None:
    if probabilities.ndim != 2 or probabilities.size(1) != num_classes:
        raise ValueError(f"probabilities must have shape [num_nodes, {num_classes}].")
    if probabilities.size(0) == 0:
        raise ValueError("probabilities cannot be empty.")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite.")
    if (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(
            probabilities.size(0),
            dtype=probabilities.dtype,
            device=probabilities.device,
        ),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError("probability rows must be nonnegative and sum to one.")


def _group_mean(values: Tensor, group_index: Tensor, num_groups: int) -> Tensor:
    if group_index.ndim != 1 or group_index.numel() != values.size(0):
        raise ValueError("group_index must identify every probability row.")
    totals = values.new_zeros((num_groups, values.size(1)))
    totals.index_add_(0, group_index, values)
    counts = torch.bincount(group_index, minlength=num_groups).to(values.dtype)
    if (counts == 0).any():
        raise ValueError("Every group must contain at least one node.")
    return totals / counts.unsqueeze(1)


class AggregationRule(nn.Module, ABC):
    fitted: Tensor

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.num_classes = num_classes
        self.register_buffer("fitted", torch.tensor(False))

    def fit(
        self,
        train_probabilities: Tensor,
        train_labels: Tensor,
        calibration_probabilities: Tensor,
        calibration_labels: Tensor,
    ) -> None:
        _validate_probabilities(train_probabilities, self.num_classes)
        _validate_probabilities(calibration_probabilities, self.num_classes)
        for name, labels, size in (
            ("training", train_labels, train_probabilities.size(0)),
            ("calibration", calibration_labels, calibration_probabilities.size(0)),
        ):
            if labels.ndim != 1 or labels.numel() != size:
                raise ValueError(f"{name} labels must identify every probability row.")
            missing = [
                cls for cls in range(self.num_classes) if not (labels == cls).any()
            ]
            if missing:
                raise ValueError(
                    f"{name} data contain no examples for classes {missing}."
                )
        self._fit(
            train_probabilities,
            train_labels,
            calibration_probabilities,
            calibration_labels,
        )
        self.fitted.fill_(True)

    def _fit(
        self,
        train_probabilities: Tensor,
        train_labels: Tensor,
        calibration_probabilities: Tensor,
        calibration_labels: Tensor,
    ) -> None:
        pass

    @abstractmethod
    def predict_many(
        self,
        probabilities: Tensor,
        group_index: Tensor,
        num_groups: int,
    ) -> Tensor: ...

    def _check_predict(self, probabilities: Tensor) -> None:
        _validate_probabilities(probabilities, self.num_classes)
        if not self.fitted:
            raise RuntimeError("The aggregation rule must be fitted before prediction.")


class KDEyMaximumLikelihood(AggregationRule):
    """KDEy-ML with optional bag-local structural importance weights."""

    def __init__(
        self,
        num_classes: int,
        bandwidth: float = 0.1,
        structural_correction: bool = False,
        correction_steps: int = 10,
        correction_restart_probability: float = 0.1,
        correction_kernel_weight: float = 0.9,
    ):
        super().__init__(num_classes)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if correction_steps < 1:
            raise ValueError("correction_steps must be at least one.")
        if not 0 <= correction_restart_probability <= 1:
            raise ValueError("correction_restart_probability must lie in [0, 1].")
        if not 0 <= correction_kernel_weight <= 1:
            raise ValueError("correction_kernel_weight must lie in [0, 1].")
        self.bandwidth = bandwidth
        self.structural_correction = structural_correction
        self.correction_steps = correction_steps
        self.correction_restart_probability = correction_restart_probability
        self.correction_kernel_weight = correction_kernel_weight
        self._centers: np.ndarray | None = None
        self._center_labels: np.ndarray | None = None

    def _fit(
        self,
        train_probabilities: Tensor,
        train_labels: Tensor,
        calibration_probabilities: Tensor,
        calibration_labels: Tensor,
    ) -> None:
        del train_probabilities, train_labels
        self._centers = calibration_probabilities.detach().cpu().double().numpy()
        self._center_labels = calibration_labels.detach().cpu().numpy()

    def _class_log_densities(
        self,
        probabilities: np.ndarray,
        center_weights: np.ndarray,
    ) -> np.ndarray:
        if self._centers is None or self._center_labels is None:
            raise RuntimeError("The aggregation rule must be fitted before prediction.")
        result = []
        scale = 2 * self.bandwidth**2
        for class_ in range(self.num_classes):
            class_mask = self._center_labels == class_
            weights = center_weights[class_mask]
            weights = weights / weights.sum()
            squared_distance = (
                (probabilities[:, None, :] - self._centers[class_mask][None, :, :])
                ** 2
            ).sum(axis=2)
            result.append(
                logsumexp(
                    -squared_distance / scale + np.log(weights)[None, :], axis=1
                )
            )
        return np.stack(result)

    def _predict_one(
        self, probabilities: Tensor, center_weights: Tensor | None = None
    ) -> Tensor:
        self._check_predict(probabilities)
        if self._centers is None:
            raise RuntimeError("The aggregation rule must be fitted before prediction.")
        if center_weights is None:
            weights = np.ones(self._centers.shape[0], dtype=np.float64)
        else:
            if center_weights.shape != (self._centers.shape[0],):
                raise ValueError("center_weights must identify every KDE center.")
            if not torch.isfinite(center_weights).all() or (center_weights < 0).any():
                raise ValueError("center_weights must be finite and nonnegative.")
            weights = center_weights.detach().cpu().double().numpy()
        log_densities = self._class_log_densities(
            probabilities.detach().cpu().double().numpy(), weights
        )
        epsilon = 1e-12

        def negative_log_likelihood(prevalence: np.ndarray) -> float:
            log_prevalence = np.log(np.clip(prevalence, epsilon, None))[:, None]
            return float(-logsumexp(log_prevalence + log_densities, axis=0).sum())

        estimate = qf.optim_minimize(negative_log_likelihood, self.num_classes)
        return torch.as_tensor(
            estimate, device=probabilities.device, dtype=probabilities.dtype
        )

    def predict_many(self, probabilities, group_index, num_groups):
        self._check_predict(probabilities)
        estimates = [
            self._predict_one(probabilities[group_index == group])
            for group in range(num_groups)
        ]
        return torch.stack(estimates)

    def predict_with_local_weights(
        self, probabilities: Tensor, calibration_weights: Tensor
    ) -> Tensor:
        """Run KDEy-ML with SIS-weighted class-conditional KDE centers."""
        return self._predict_one(probabilities, calibration_weights)


class SLD(AggregationRule):
    """Saerens-Latinne-Decaestecker EM prior-shift adjustment."""

    train_prevalence: Tensor

    def __init__(
        self,
        num_classes: int,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        epsilon: float = 1e-4,
    ):
        super().__init__(num_classes)
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one.")
        if tolerance <= 0 or epsilon <= 0:
            raise ValueError("tolerance and epsilon must be positive.")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.register_buffer(
            "train_prevalence", torch.full((num_classes,), 1 / num_classes)
        )

    def _fit(
        self,
        train_probabilities: Tensor,
        train_labels: Tensor,
        calibration_probabilities: Tensor,
        calibration_labels: Tensor,
    ) -> None:
        del train_probabilities, calibration_probabilities, calibration_labels
        prevalence = torch.bincount(train_labels, minlength=self.num_classes).to(
            dtype=self.train_prevalence.dtype
        )
        self.train_prevalence.copy_(prevalence / prevalence.sum())

    def predict_many(self, probabilities, group_index, num_groups):
        self._check_predict(probabilities)
        estimate = self.train_prevalence.expand(num_groups, -1).clone()
        train_prevalence = self.train_prevalence.clamp_min(self.epsilon)
        active = torch.ones(num_groups, dtype=torch.bool, device=probabilities.device)
        for iteration in range(self.max_iterations):
            ratio = estimate / train_prevalence
            adjusted = probabilities * ratio[group_index]
            adjusted = adjusted / adjusted.sum(dim=1, keepdim=True).clamp_min(
                self.epsilon
            )
            candidate = _group_mean(adjusted, group_index, num_groups)
            converged = (
                (candidate - estimate).abs().mean(dim=1) < self.tolerance
                if iteration > 10
                else torch.zeros_like(active)
            )
            estimate = torch.where(active.unsqueeze(1), candidate, estimate)
            active &= ~converged
            if not active.any():
                break
        return estimate
