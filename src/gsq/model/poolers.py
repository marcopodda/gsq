import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import MeanAggregation


class DeepSetsPooler(nn.Module):
    """Standard DeepSets over the mean of transformed node representations."""

    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        self.channels = channels
        self.local_nn = self._make_mlp(channels)
        self.global_nn = self._make_mlp(channels)
        self.mean = MeanAggregation()

    @staticmethod
    def _make_mlp(channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if x.ndim != 2 or x.size(1) != self.channels:
            raise ValueError(f"x must have shape [num_nodes, {self.channels}].")
        if x.size(0) == 0:
            raise ValueError("x must contain at least one node.")
        if batch.ndim != 1 or batch.numel() != x.size(0):
            raise ValueError("batch must assign every node to one group.")
        if batch.dtype != torch.long:
            raise TypeError("batch must use torch.long indices.")

        learned_summary = self.mean(self.local_nn(x), batch)
        return self.global_nn(learned_summary)

    def forward_rectangular(self, x: Tensor) -> Tensor:
        """Pool fixed-size groups without constructing flattened group indices."""
        if x.ndim != 3 or x.size(2) != self.channels:
            raise ValueError(
                f"x must have shape [num_groups, group_size, {self.channels}]."
            )
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("x must contain at least one non-empty group.")
        return self.global_nn(self.local_nn(x).mean(dim=1))


class GMNetPooler(nn.Module):
    """Gaussian latent-space bag representation from GMNet.

    Each view maps node representations to a small latent space, evaluates a
    bank of learned full-covariance Gaussian densities, and averages those
    likelihoods over the bag. The concatenated density summaries are projected
    back to ``channels`` so the pooler remains interchangeable with mean and
    DeepSets pooling.

    Unlike the reference fixed-size implementation, this module accepts both
    grouped variable-size bags and rectangular batches. Positive-definite
    covariance matrices are parameterized by their Cholesky factors instead of
    being repaired by an external parametrization package.
    """

    # Bound the dominant likelihood workspace independently of launcher config.
    # With APP bags of 500 nodes this limits a forward chunk to 16 bags.
    max_nodes_per_chunk = 8_000
    # A training step contains many bag chunks. Without checkpointing, PyTorch
    # retains every branch's Gaussian activations for every chunk until the
    # single Lightning backward pass. QuaPy avoids this by taking an optimizer
    # step every 16 bags; checkpointing gives this integration the same bounded
    # activation footprint without changing its one-step-per-epoch semantics.
    checkpoint_chunks = True

    def __init__(
        self,
        channels: int,
        num_views: int = 9,
        num_gaussians: int = 100,
        latent_channels: int = 5,
        hidden_channels: int = 50,
        dropout: float = 0.5,
        covariance_epsilon: float = 1e-4,
    ):
        super().__init__()
        if (
            min(
                channels,
                num_views,
                num_gaussians,
                latent_channels,
                hidden_channels,
            )
            < 1
        ):
            raise ValueError("GMNet dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if covariance_epsilon <= 0:
            raise ValueError("covariance_epsilon must be positive.")

        self.channels = channels
        self.num_views = num_views
        self.num_gaussians = num_gaussians
        self.latent_channels = latent_channels
        self.covariance_epsilon = covariance_epsilon
        self.views = nn.ModuleList(
            nn.Sequential(
                nn.Linear(channels, hidden_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, latent_channels),
                nn.Sigmoid(),
            )
            for _ in range(num_views)
        )
        self.centers = nn.Parameter(
            torch.rand(num_views, num_gaussians, latent_channels)
        )
        initial_scale = torch.tensor(0.1 - covariance_epsilon).clamp_min(1e-6)
        initial_diagonal = torch.log(torch.expm1(initial_scale))
        raw_cholesky = torch.zeros(
            num_views,
            num_gaussians,
            latent_channels,
            latent_channels,
        )
        raw_cholesky.diagonal(dim1=-2, dim2=-1).fill_(initial_diagonal)
        self.raw_cholesky = nn.Parameter(raw_cholesky)
        self.output = nn.Linear(num_views * num_gaussians, channels)
        self.mean = MeanAggregation()

    def _validate_features(self, x: Tensor) -> None:
        if x.ndim != 2 or x.size(1) != self.channels:
            raise ValueError(f"x must have shape [num_nodes, {self.channels}].")
        if x.size(0) == 0:
            raise ValueError("x must contain at least one node.")

    def _cholesky(self, view: int) -> Tensor:
        raw = self.raw_cholesky[view].float()
        lower = raw.tril(diagonal=-1)
        diagonal = F.softplus(raw.diagonal(dim1=-2, dim2=-1))
        diagonal = diagonal + self.covariance_epsilon
        return lower + torch.diag_embed(diagonal)

    def _likelihoods(self, x: Tensor, view: int) -> Tensor:
        # Gaussian linear algebra stays in float32: determinants and triangular
        # solves are fragile in mixed precision even when the final loss is not.
        with torch.autocast(device_type=x.device.type, enabled=False):
            latent = self.views[view](x.float())
            difference = latent.unsqueeze(1) - self.centers[view].float().unsqueeze(0)
            cholesky = self._cholesky(view)
            # Differentiate through a solve whose right-hand side is only the
            # tiny latent identity, then apply the resulting inverse factors
            # to nodes with an einsum. Passing every node as a solve RHS keeps
            # the forward compact but makes solve_triangular's CUDA backward
            # allocate a workspace that scales catastrophically with the
            # number of nodes (about 39 GiB for a 16 x 500 APP chunk).
            identity = torch.eye(
                self.latent_channels,
                dtype=cholesky.dtype,
                device=cholesky.device,
            ).expand(self.num_gaussians, -1, -1)
            inverse_cholesky = torch.linalg.solve_triangular(
                cholesky,
                identity,
                upper=False,
            )
            whitened = torch.einsum(
                "gij,ngj->ngi",
                inverse_cholesky,
                difference,
            )
            mahalanobis = whitened.square().sum(dim=-1)
            log_determinant = 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            normalization = self.latent_channels * torch.log(
                latent.new_tensor(2 * torch.pi)
            )
            log_density = -0.5 * (
                mahalanobis + log_determinant.unsqueeze(0) + normalization
            )
            # Values below this threshold carry no useful signal in float32;
            # clamping prevents exact underflow and preserves finite gradients.
            return log_density.clamp_min(-80).exp()

    def _project(self, summaries: list[Tensor], dtype: torch.dtype) -> Tensor:
        representation = torch.cat(summaries, dim=-1)
        return self.output(representation).to(dtype=dtype)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        self._validate_features(x)
        if batch.ndim != 1 or batch.numel() != x.size(0):
            raise ValueError("batch must assign every node to one group.")
        if batch.dtype != torch.long:
            raise TypeError("batch must use torch.long indices.")
        if (batch < 0).any():
            raise ValueError("batch indices must be nonnegative.")
        summaries = [
            self.mean(self._likelihoods(x, view), batch)
            for view in range(self.num_views)
        ]
        return self._project(summaries, x.dtype)

    def forward_rectangular(self, x: Tensor) -> Tensor:
        """Pool fixed-size bags without materializing flattened group indices."""
        if x.ndim != 3 or x.size(2) != self.channels:
            raise ValueError(
                f"x must have shape [num_groups, group_size, {self.channels}]."
            )
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("x must contain at least one non-empty group.")
        flat = x.flatten(0, 1)
        summaries = [
            self._likelihoods(flat, view)
            .unflatten(0, (x.size(0), x.size(1)))
            .mean(dim=1)
            for view in range(self.num_views)
        ]
        return self._project(summaries, x.dtype)


