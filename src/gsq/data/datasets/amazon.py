import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import gsq

from .base import AddSplitsTransform, BaseDataset


class AmazonRatings(BaseDataset):
    """Amazon product ratings graph with GSQ's five-fold splits."""

    url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"

    def __init__(
        self,
        root: str = gsq.DATA_DIR.as_posix(),
        name: str = "amazon_ratings",
        fold_index: int = 0,
    ):
        self.name = name
        transform = AddSplitsTransform(root, name, fold_index)
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.name}.npz"]

    def process(self) -> None:
        raw = np.load(self.raw_paths[0], "r")
        x = torch.from_numpy(raw["node_features"]).float()
        y = torch.from_numpy(raw["node_labels"]).squeeze().long()
        edge_index = torch.from_numpy(raw["edges"]).t().long()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0)).contiguous()
        self.save([Data(x=x, edge_index=edge_index, y=y)], self.processed_paths[0])
