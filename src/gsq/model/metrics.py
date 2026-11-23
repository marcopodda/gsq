import torch
from torch import Tensor
from torchmetrics import Metric


class MeanRAE(Metric):
    """Mean relative absolute error over samples and classes."""

    full_state_update = False

    def __init__(self, smooth: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 < smooth < 0.5:
            raise ValueError("smooth must be in the open interval (0, 0.5).")
        self.smooth = smooth
        self.error_sum: Tensor
        self.count: Tensor
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target shapes differ: {prediction.shape} != {target.shape}."
            )
        target = target.float() + self.smooth
        prediction = prediction.detach().float() + self.smooth
        target = target / target.sum(dim=-1, keepdim=True)
        prediction = prediction / prediction.sum(dim=-1, keepdim=True)
        self.error_sum += ((prediction - target).abs() / target).sum()
        self.count += target.numel()

    def compute(self) -> Tensor:
        if self.count == 0:
            return self.error_sum.new_tensor(float("nan"))
        return self.error_sum / self.count
