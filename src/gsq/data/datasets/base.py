from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset, download_url


class AddSplitsTransform:
    def __init__(self, root: str, name: str, fold_index: int):
        import gsq

        self.splits_dir = gsq.SPLITS_DIR / name
        self.fold_index = fold_index
        self.train_masks = self._load("training")
        self.val_masks = self._load("validation")
        self.cal_masks = self._load("calibration")
        self.test_masks = self._load("test")

        num_folds = self.train_masks.size(0)
        if not 0 <= fold_index < num_folds:
            raise ValueError(f"fold_index must be between 0 and {num_folds - 1}")
        masks = {
            "training": self.train_masks,
            "calibration": self.cal_masks,
            "validation": self.val_masks,
            "test": self.test_masks,
        }
        expected_shape = self.train_masks.shape
        for split_name, mask in masks.items():
            if mask.dtype != torch.bool or mask.ndim != 2:
                raise TypeError(
                    f"{split_name} masks must be a two-dimensional bool tensor."
                )
            if mask.shape != expected_shape:
                raise ValueError(
                    f"{split_name} masks have shape {tuple(mask.shape)}, expected "
                    f"{tuple(expected_shape)}."
                )
        coverage = torch.stack(list(masks.values())).sum(dim=0)
        if not torch.equal(coverage, torch.ones_like(coverage)):
            raise ValueError("Split masks must be disjoint and cover every node once.")

    def _load(self, split: str) -> torch.Tensor:
        return torch.load(
            self.splits_dir / f"{split}_masks.pt",
            map_location="cpu",
            weights_only=True,
        )

    def __call__(self, data):
        if data.num_nodes != self.train_masks.size(1):
            raise ValueError(
                f"Split {self.splits_dir.name!r} has {self.train_masks.size(1)} "
                f"nodes, but the dataset has {data.num_nodes}"
            )
        data.y = data.y.long()
        data.train_mask = (
            self.train_masks[self.fold_index] | self.cal_masks[self.fold_index]
        )
        data.val_mask = self.val_masks[self.fold_index]
        data.cal_mask = self.cal_masks[self.fold_index]
        data.test_mask = self.test_masks[self.fold_index]
        return data


class BaseDataset(InMemoryDataset):
    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / self.name / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / self.name / "processed")

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self) -> None:
        for name in self.raw_file_names:
            download_url(f"{self.url}/{name}", self.raw_dir)

    @property
    def num_nodes(self) -> int:
        num_nodes = self.get(0).num_nodes
        if num_nodes is None:
            raise ValueError("The processed graph does not define its number of nodes.")
        return num_nodes
