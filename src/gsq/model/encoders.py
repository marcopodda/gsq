from itertools import pairwise
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import APPNP, GCNConv

EncoderType = Literal["gcn", "mlp"]


def _sparse_graph_spectral_norm(
    edge_index: Tensor,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float:
    """Estimate the adjacency spectral norm without densifying the graph.

    Power iteration is applied to ``A.T @ A`` so this computes the largest
    singular value for directed as well as undirected graphs. The iteration is
    deliberately performed in float64 on CPU: initialization happens only once,
    and this keeps the reservoir scaling deterministic across CUDA workers.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long indices.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if edge_index.numel() == 0:
        return 0.0

    edges = edge_index.detach().to(device="cpu")
    source, target = edges
    num_nodes = int(edges.max()) + 1
    right = torch.ones(num_nodes, dtype=torch.float64)
    right /= torch.linalg.vector_norm(right)

    previous = 0.0
    for _ in range(max_iterations):
        left = torch.zeros_like(right)
        left.index_add_(0, target, right[source])
        left_norm = torch.linalg.vector_norm(left)
        if left_norm == 0:
            return 0.0
        left /= left_norm

        next_right = torch.zeros_like(right)
        next_right.index_add_(0, source, left[target])
        next_norm = torch.linalg.vector_norm(next_right)
        if next_norm == 0:
            return 0.0
        right = next_right / next_norm

        projected = torch.zeros_like(right)
        projected.index_add_(0, target, right[source])
        estimate = float(torch.linalg.vector_norm(projected))
        if abs(estimate - previous) <= tolerance * max(1.0, estimate):
            return estimate
        previous = estimate
    return previous


class NodeEncoder(nn.Module):
    """A small stack of node-wise or message-passing layers."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        encoder_type: EncoderType,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.dropout = dropout

        self.input = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [self._make_layer(encoder_type, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )

    @staticmethod
    def _make_layer(encoder_type: EncoderType, channels: int) -> nn.Module:
        if encoder_type == "gcn":
            return GCNConv(channels, channels)  # ty: ignore[abstract-class-instantiation]
        if encoder_type == "mlp":
            return nn.Linear(channels, channels)
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x0 = F.relu(self.input(x))
        h = x0
        for layer, norm in zip(self.layers, self.norms):
            if self.encoder_type == "mlp":
                h = layer(h)
            else:
                h = layer(h, edge_index)
            h = F.dropout(F.relu(norm(h)), self.dropout, self.training)
        return h


class APPNPEncoder(nn.Module):
    """Paper-style MLP followed by approximate personalized propagation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        propagation_steps: int = 10,
        restart_probability: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one.")
        if propagation_steps < 1:
            raise ValueError("propagation_steps must be at least one.")
        if not 0 < restart_probability <= 1:
            raise ValueError("restart_probability must lie in (0, 1].")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.propagation_steps = propagation_steps
        self.restart_probability = restart_probability
        self.dropout = dropout
        dimensions = [in_channels, *([hidden_channels] * num_layers)]
        self.layers = nn.ModuleList(
            nn.Linear(source, target) for source, target in pairwise(dimensions)
        )
        self.propagation = APPNP(  # ty: ignore[abstract-class-instantiation]
            K=propagation_steps,
            alpha=restart_probability,
            dropout=0.0,
            add_self_loops=True,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        return self.propagation(x, edge_index)


class _ReservoirLayer(nn.Module):
    """Dependency-free equivalent of GraphESN's static convolution layer."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.input_weight = nn.Parameter(
            torch.empty(hidden_channels, in_channels), requires_grad=False
        )
        self.recurrent_weight = nn.Parameter(
            torch.empty(hidden_channels, hidden_channels), requires_grad=False
        )

    def forward(self, x: Tensor, state: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        neighbours = torch.zeros_like(state)
        neighbours.index_add_(0, target, state[source])
        return torch.tanh(
            F.linear(x, self.input_weight)
            + F.linear(neighbours, self.recurrent_weight)
        )


class _StaticGraphReservoir(nn.Module):
    """The GraphESN update used in the experiments, without torch-sparse."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        max_iterations: int,
    ):
        super().__init__()
        channels = [in_channels, *([hidden_channels] * num_layers)]
        self.layers = nn.ModuleList(
            _ReservoirLayer(left, right)
            for left, right in pairwise(channels)
        )
        self.max_iterations = max_iterations

    def forward(self, edge_index: Tensor, x: Tensor) -> Tensor:
        for raw_layer in self.layers:
            layer = cast(_ReservoirLayer, raw_layer)
            state = x.new_zeros((x.size(0), layer.recurrent_weight.size(0)))
            # GraphESN checks its iteration limit after computing the update;
            # preserve that convention for experiment reproducibility.
            for _ in range(self.max_iterations + 1):
                next_state = layer(x, state, edge_index)
                if torch.linalg.vector_norm(state - next_state) < 1e-6:
                    state = next_state
                    break
                state = next_state
            x = state
        return x


class ReservoirEncoder(nn.Module):
    """A lazily initialized, frozen graph echo-state network encoder."""

    initialized: Tensor

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        recurrent_scale: float,
        input_scale: float,
        num_layers: int = 1,
        max_iterations: int = 30,
        random_seed: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.recurrent_scale = recurrent_scale
        self.input_scale = input_scale
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.reservoir = _StaticGraphReservoir(
            num_layers=num_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            max_iterations=max_iterations,
        )
        self.reservoir.requires_grad_(False)
        self.register_buffer("initialized", torch.tensor(False))

    def _initialize(self, edge_index: Tensor) -> None:
        spectral_norm = _sparse_graph_spectral_norm(edge_index)
        if spectral_norm <= 0:
            raise ValueError(
                "ReservoirEncoder requires a graph with non-zero spectral norm."
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            for raw_layer in self.reservoir.layers:
                layer = cast(_ReservoirLayer, raw_layer)
                recurrent = torch.empty_like(layer.recurrent_weight).uniform_(-1, 1)
                radius = torch.linalg.eigvals(recurrent).abs().max()
                recurrent.mul_(self.recurrent_scale / spectral_norm / radius)
                inputs = torch.empty_like(layer.input_weight).uniform_(-1, 1)
                inputs.mul_(self.input_scale)
                layer.recurrent_weight.copy_(recurrent)
                layer.input_weight.copy_(inputs)
        self.initialized.fill_(True)

    def initialize(self, edge_index: Tensor) -> None:
        """Initialize frozen weights once, independently of propagation device."""
        if not self.initialized:
            self._initialize(edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        self.initialize(edge_index)
        return self.reservoir(edge_index, x)


class PCAReservoirEncoder(nn.Module):
    """A zero-parameter handle for precomputed PCA reservoir embeddings.

    The raw reservoir and PCA projection exist only during cache construction.
    Training and evaluation load the materialized node embeddings directly.
    """

    def __init__(
        self,
        in_channels: int,
        reservoir_channels: int,
        hidden_channels: int,
        recurrent_scale: float,
        input_scale: float,
        num_layers: int = 1,
        max_iterations: int = 30,
        random_seed: int = 0,
        pca_oversampling: int = 16,
        pca_iterations: int = 4,
    ):
        if hidden_channels < 1 or hidden_channels > reservoir_channels:
            raise ValueError("hidden_channels must lie in [1, reservoir_channels].")
        if pca_oversampling < 0:
            raise ValueError("pca_oversampling must be nonnegative.")
        if pca_iterations < 0:
            raise ValueError("pca_iterations must be nonnegative.")
        super().__init__()
        self.in_channels = in_channels
        self.reservoir_channels = reservoir_channels
        self.hidden_channels = hidden_channels
        self.recurrent_scale = recurrent_scale
        self.input_scale = input_scale
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.pca_oversampling = pca_oversampling
        self.pca_iterations = pca_iterations

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        raise RuntimeError(
            "PCAReservoirEncoder only consumes precomputed PCA node embeddings. "
            "Run phase 00 to create the required cache."
        )
