from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from quapy.data import LabelledCollection
from quapy.protocol import OnLabelledCollectionProtocol


@dataclass(frozen=True)
class PreparedStructuralGraph:
    """Immutable full-graph adjacency shared by structural protocol samples."""

    neighbors: tuple[NDArray[np.int64], ...]

    @property
    def num_nodes(self) -> int:
        return len(self.neighbors)


class StructuralProtocol(OnLabelledCollectionProtocol):
    """Base protocol for class-stratified, graph-local test samples."""

    def __init__(
        self,
        data: LabelledCollection,
        edge_index: NDArray[np.int64],
        *,
        sample_size: int = 100,
        seeds_per_class: int | None = 10,
        num_seeds: int | None = None,
        random_state: int | None = 0,
        return_type: str = "sample_prev",
    ) -> None:
        super().__init__(random_state)
        if sample_size < 1:
            raise ValueError("sample_size must be at least one.")
        if seeds_per_class is not None and seeds_per_class < 1:
            raise ValueError("seeds_per_class must be at least one.")
        if num_seeds is not None and num_seeds < 1:
            raise ValueError("num_seeds must be at least one.")
        if (seeds_per_class is None) == (num_seeds is None):
            raise ValueError("Define exactly one of seeds_per_class and num_seeds.")

        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edges.size and (np.any(edges < 0) or np.any(edges >= len(data))):
            raise ValueError("edge_index contains nodes outside the collection.")

        self.data = data
        self.edge_index = edges
        self.sample_size = sample_size
        self.seeds_per_class = seeds_per_class
        self.num_seeds = num_seeds
        self.collator = OnLabelledCollectionProtocol.get_collator(return_type)

    @staticmethod
    def _adjacency_lists(
        num_nodes: int, edge_index: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], ...]:
        neighbors = [set() for _ in range(num_nodes)]
        for source, target in edge_index.T:
            neighbors[int(source)].add(int(target))
            neighbors[int(target)].add(int(source))
        return tuple(np.fromiter(sorted(row), dtype=np.int64) for row in neighbors)

    @classmethod
    def prepare_full_graph(
        cls,
        num_nodes: int,
        edge_index: NDArray[np.int64],
    ) -> PreparedStructuralGraph:
        """Validate and materialize adjacency once for repeated bag sampling."""
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative.")
        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edges.size and (np.any(edges < 0) or np.any(edges >= num_nodes)):
            raise ValueError("full_edge_index contains nodes outside the graph.")
        return PreparedStructuralGraph(cls._adjacency_lists(num_nodes, edges))

    def _seed_nodes(self) -> NDArray[np.int64]:
        if self.num_seeds is not None:
            if self.num_seeds > len(self.data):
                raise ValueError(
                    f"Requested {self.num_seeds} seeds from {len(self.data)} nodes."
                )
            return np.random.choice(
                len(self.data), size=self.num_seeds, replace=False
            ).astype(np.int64)

        assert self.seeds_per_class is not None
        selected = []
        for class_ in self.data.classes_:
            candidates = np.flatnonzero(self.data.labels == class_)
            if candidates.size < self.seeds_per_class:
                raise ValueError(
                    f"Class {class_!r} has {candidates.size} nodes, fewer than the "
                    f"requested {self.seeds_per_class} seeds."
                )
            np.random.shuffle(candidates)
            selected.extend(candidates[: self.seeds_per_class])
        return np.asarray(selected, dtype=np.int64)

    def samples_parameters(self) -> list[NDArray[np.int64]]:
        return [self._sample_from_seed(int(seed)) for seed in self._seed_nodes()]

    def total(self) -> int:
        if self.num_seeds is not None:
            return self.num_seeds
        assert self.seeds_per_class is not None
        return self.seeds_per_class * self.data.n_classes

    def _sample_from_seed(self, seed: int) -> NDArray[np.int64]:
        raise NotImplementedError


class BFPProtocol(StructuralProtocol):
    """Samples unique eligible endpoints reached through the complete graph."""

    def __init__(
        self,
        *args,
        max_depth: int = 10,
        full_edge_index: NDArray[np.int64] | None = None,
        evaluation_nodes: NDArray[np.int64] | None = None,
        prepared_full_graph: PreparedStructuralGraph | None = None,
        **kwargs,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be nonnegative.")
        self.max_depth = max_depth
        super().__init__(*args, **kwargs)

        self.evaluation_nodes = np.asarray(
            np.arange(len(self.data)) if evaluation_nodes is None else evaluation_nodes,
            dtype=np.int64,
        )
        if self.evaluation_nodes.shape != (len(self.data),):
            raise ValueError("evaluation_nodes must map every collection node.")
        if np.unique(self.evaluation_nodes).size != self.evaluation_nodes.size:
            raise ValueError("evaluation_nodes must contain unique full-graph nodes.")
        if self.evaluation_nodes.size and np.any(self.evaluation_nodes < 0):
            raise ValueError("evaluation_nodes must be nonnegative.")

        evaluation_node_count = (
            int(np.max(self.evaluation_nodes)) + 1
            if self.evaluation_nodes.size
            else 0
        )
        if prepared_full_graph is None:
            full_edges = np.asarray(
                self.edge_index if full_edge_index is None else full_edge_index,
                dtype=np.int64,
            )
            if full_edges.ndim != 2 or full_edges.shape[0] != 2:
                raise ValueError("full_edge_index must have shape [2, num_edges].")
            edge_nodes = int(np.max(full_edges)) + 1 if full_edges.size else 0
            self.num_full_nodes = max(edge_nodes, evaluation_node_count)
            if full_edges.size and np.any(full_edges < 0):
                raise ValueError("full_edge_index contains negative node indices.")
            self._full_neighbors = self._adjacency_lists(
                self.num_full_nodes, full_edges
            )
        else:
            if prepared_full_graph.num_nodes < evaluation_node_count:
                raise ValueError(
                    "prepared_full_graph does not cover every full-graph node."
                )
            self.num_full_nodes = prepared_full_graph.num_nodes
            self._full_neighbors = prepared_full_graph.neighbors
        self._local_index = np.full(self.num_full_nodes, -1, dtype=np.int64)
        self._local_index[self.evaluation_nodes] = np.arange(len(self.data))
        self.sample_sizes = np.empty(0, dtype=np.int64)

    def samples_parameters(self) -> list[NDArray[np.int64]]:
        samples = super().samples_parameters()
        self.sample_sizes = np.asarray([sample.size for sample in samples])
        return samples

    def _sample_from_seed(self, seed: int) -> NDArray[np.int64]:
        full_seed = int(self.evaluation_nodes[seed])
        queue = deque([(full_seed, 0)])
        visited = {full_seed}
        sample = []

        while queue and len(sample) < self.sample_size:
            node, depth = queue.popleft()
            local_node = int(self._local_index[node])
            if local_node >= 0:
                sample.append(local_node)
            if depth == self.max_depth:
                continue
            for neighbor in self._full_neighbors[node]:
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return np.asarray(sample, dtype=np.int64)


class RWPProtocol(StructuralProtocol):
    """Collect unique eligible nodes during a full-graph walk with restart."""

    def __init__(
        self,
        *args,
        restart_probability: float = 0.1,
        max_steps: int = 10_000,
        full_edge_index: NDArray[np.int64] | None = None,
        evaluation_nodes: NDArray[np.int64] | None = None,
        prepared_full_graph: PreparedStructuralGraph | None = None,
        **kwargs,
    ) -> None:
        if not 0 <= restart_probability <= 1:
            raise ValueError("restart_probability must lie in [0, 1].")
        if max_steps < 1:
            raise ValueError("max_steps must be at least one.")
        self.restart_probability = restart_probability
        self.max_steps = max_steps
        super().__init__(*args, **kwargs)
        self.evaluation_nodes = np.asarray(
            np.arange(len(self.data)) if evaluation_nodes is None else evaluation_nodes,
            dtype=np.int64,
        )
        if self.evaluation_nodes.shape != (len(self.data),):
            raise ValueError("evaluation_nodes must map every collection node.")
        if np.unique(self.evaluation_nodes).size != self.evaluation_nodes.size:
            raise ValueError("evaluation_nodes must contain unique full-graph nodes.")
        if self.evaluation_nodes.size and np.any(self.evaluation_nodes < 0):
            raise ValueError("evaluation_nodes must be nonnegative.")
        evaluation_node_count = (
            int(np.max(self.evaluation_nodes)) + 1
            if self.evaluation_nodes.size
            else 0
        )
        if prepared_full_graph is None:
            full_edges = np.asarray(
                self.edge_index if full_edge_index is None else full_edge_index,
                dtype=np.int64,
            )
            if full_edges.ndim != 2 or full_edges.shape[0] != 2:
                raise ValueError("full_edge_index must have shape [2, num_edges].")
            if full_edges.size and np.any(full_edges < 0):
                raise ValueError("full_edge_index contains negative node indices.")
            edge_nodes = int(np.max(full_edges)) + 1 if full_edges.size else 0
            self.num_full_nodes = max(edge_nodes, evaluation_node_count)
            self._full_neighbors = self._adjacency_lists(
                self.num_full_nodes, full_edges
            )
        else:
            if prepared_full_graph.num_nodes < evaluation_node_count:
                raise ValueError(
                    "prepared_full_graph does not cover every full-graph node."
                )
            self.num_full_nodes = prepared_full_graph.num_nodes
            self._full_neighbors = prepared_full_graph.neighbors
        self._local_index = np.full(self.num_full_nodes, -1, dtype=np.int64)
        self._local_index[self.evaluation_nodes] = np.arange(len(self.data))
        self.sample_sizes = np.empty(0, dtype=np.int64)
        self.traversal_steps = np.empty(0, dtype=np.int64)

    def _crawl_from_seed(self, seed: int) -> tuple[NDArray[np.int64], int]:
        full_seed = int(self.evaluation_nodes[seed])
        node = full_seed
        selected: set[int] = set()
        sample: list[int] = []

        for step in range(self.max_steps + 1):
            local_node = int(self._local_index[node])
            if local_node >= 0 and local_node not in selected:
                selected.add(local_node)
                sample.append(local_node)
                if len(sample) == self.sample_size:
                    return np.asarray(sample, dtype=np.int64), step
            if step == self.max_steps:
                break
            neighbors = self._full_neighbors[node]
            if neighbors.size == 0 or np.random.random() < self.restart_probability:
                node = full_seed
            else:
                node = int(np.random.choice(neighbors))
        return np.asarray(sample, dtype=np.int64), self.max_steps

    def samples_parameters(self) -> list[NDArray[np.int64]]:
        crawls = [self._crawl_from_seed(int(seed)) for seed in self._seed_nodes()]
        samples = [sample for sample, _ in crawls]
        self.sample_sizes = np.asarray([sample.size for sample in samples])
        self.traversal_steps = np.asarray([steps for _, steps in crawls])
        return samples

    def _sample_from_seed(self, seed: int) -> NDArray[np.int64]:
        return self._crawl_from_seed(seed)[0]
