import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MSEHead(nn.Module):
    """Linear softmax prevalence predictor trained with mean squared error."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)

    def predict(self, x: Tensor) -> Tensor:
        return self.linear(x).softmax(dim=-1)

    def forward(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        prediction = self.predict(x)
        return F.mse_loss(prediction, target), prediction


class DirichletHead(nn.Module):
    """Regress prevalence as the mean of a learned Dirichlet distribution."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        concentration_penalty: float = 1e-3,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_classes)
        self.concentration_penalty = concentration_penalty
        self.epsilon = epsilon

    def concentrations(self, x: Tensor) -> Tensor:
        return F.softplus(self.linear(x)) + self.epsilon

    def predict(self, x: Tensor) -> Tensor:
        concentration = self.concentrations(x)
        return concentration / concentration.sum(dim=-1, keepdim=True)

    def forward(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        concentration = self.concentrations(x)
        target = target.clamp_min(self.epsilon)
        target = target / target.sum(dim=-1, keepdim=True)
        loss = -torch.distributions.Dirichlet(concentration).log_prob(target)
        loss += self.concentration_penalty * concentration.sum(dim=-1).log().square()
        prediction = concentration / concentration.sum(dim=-1, keepdim=True)
        return loss.mean(), prediction
