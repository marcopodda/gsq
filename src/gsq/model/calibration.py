import torch
from torch import Tensor, nn

CALIBRATION_MODES = ("none", "scalar", "classwise")


class PrevalenceCalibrator(nn.Module):
    """Temperature or vector scaling for bag-level prevalence predictions."""

    def __init__(self, num_classes: int, classwise: bool = True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")
        self.classwise = classwise
        parameter_count = num_classes if classwise else 1
        self.log_temperature = nn.Parameter(
            torch.zeros(parameter_count), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(parameter_count), requires_grad=False)

    def regularization_loss(self) -> Tensor:
        """Shrink temperatures toward one and class deviations toward zero."""
        mean_log_temperature = self.log_temperature.mean()
        centered_log_temperature = self.log_temperature - mean_log_temperature
        centered_bias = self.bias - self.bias.mean()
        return (
            mean_log_temperature.square()
            + centered_log_temperature.square().mean()
            + centered_bias.square().mean()
        )

    def forward(self, prevalence: Tensor) -> Tensor:
        positive = prevalence > 0
        tiny = torch.finfo(prevalence.dtype).tiny
        logits = prevalence.clamp_min(tiny).log()
        bias = self.bias - self.bias.mean()
        scaled = logits / self.log_temperature.exp() + bias
        return scaled.masked_fill(~positive, -torch.inf).softmax(dim=-1)

    def fit(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        max_steps: int = 100,
        regularization: float = 1e-3,
    ) -> float:
        if prediction.shape != target.shape or prediction.ndim != 2:
            raise ValueError("prediction and target must share shape [bags, classes].")
        if max_steps < 1:
            raise ValueError("max_steps must be at least one.")
        if regularization < 0:
            raise ValueError("regularization must be nonnegative.")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("prediction and target must be finite.")
        if (prediction < 0).any() or (target < 0).any():
            raise ValueError("prediction and target must be nonnegative.")
        for name, values in (("prediction", prediction), ("target", target)):
            if not torch.allclose(
                values.sum(dim=1),
                torch.ones(values.size(0), device=values.device, dtype=values.dtype),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise ValueError(f"Every {name} row must sum to one.")

        # Tensors produced under inference_mode cannot be retained for LBFGS
        # backward, even after detach. Cloning outside that context creates
        # ordinary tensors. Calibration stays in float32 under mixed precision.
        prediction = prediction.detach().float().clone()
        target = target.detach().float().clone()
        initial_temperature = self.log_temperature.detach().clone()
        initial_bias = self.bias.detach().clone()
        self.log_temperature.requires_grad_(True)
        self.bias.requires_grad_(self.classwise)
        parameters = [self.log_temperature]
        if self.classwise:
            parameters.append(self.bias)
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=0.1,
            max_iter=max_steps,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            calibrated = self(prediction)
            loss = -(target * calibrated.clamp_min(1e-12).log()).sum(dim=-1).mean()
            loss = loss + regularization * self.regularization_loss()
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except BaseException:
            with torch.no_grad():
                self.log_temperature.copy_(initial_temperature)
                self.bias.copy_(initial_bias)
            raise
        finally:
            self.log_temperature.requires_grad_(False)
            self.bias.requires_grad_(False)
        if not (
            torch.isfinite(self.log_temperature).all()
            and torch.isfinite(self.bias).all()
        ):
            with torch.no_grad():
                self.log_temperature.copy_(initial_temperature)
                self.bias.copy_(initial_bias)
            raise FloatingPointError("Calibration produced non-finite parameters.")
        with torch.no_grad():
            calibrated = self(prediction)
            loss = -(target * calibrated.clamp_min(1e-12).log()).sum(dim=-1).mean()
            loss = loss + regularization * self.regularization_loss()
        if not torch.isfinite(loss):
            with torch.no_grad():
                self.log_temperature.copy_(initial_temperature)
                self.bias.copy_(initial_bias)
            raise FloatingPointError("Calibration produced a non-finite objective.")
        return float(loss)


