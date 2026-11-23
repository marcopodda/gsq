import torch_geometric.transforms as T
from torch_geometric.io import read_planetoid_data

import gsq

from .base import AddSplitsTransform, BaseDataset


class BinaryCoraTransform:
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        data.y = (data.y == self.target).long()
        return data


def _compose_transforms(root, name, fold_index, target):
    if target is None:
        return T.Compose([AddSplitsTransform(root, name, fold_index)])
    return T.Compose(
        [
            AddSplitsTransform(root, name, fold_index),
            BinaryCoraTransform(target),
        ]
    )


class Cora(BaseDataset):
    """Loads Cora Dataset with custom splits."""

    url = "https://github.com/kimiyoung/planetoid/raw/master/data"

    def __init__(
        self,
        root: str = gsq.DATA_DIR.as_posix(),
        name: str = "cora",
        target: int | None = None,
        fold_index: int = 0,
    ):
        self.name = name
        self.fold_index = fold_index
        self.target = target

        transform = _compose_transforms(root, self.full_name, fold_index, target)
        super().__init__(root, transform=transform)
        self.load(self.processed_paths[0])

    @property
    def full_name(self) -> str:
        if self.target is None:
            return self.name
        return f"cora-{self.target}-vs-all"

    @property
    def raw_file_names(self) -> list[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name}.{name}" for name in names]

    def process(self) -> None:
        data = read_planetoid_data(self.raw_dir, self.name)
        self.save([data], self.processed_paths[0])
