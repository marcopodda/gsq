import hydra
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader, NeighborLoader


class QuantificationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        loader: DictConfig,
    ) -> None:
        super().__init__()

        self.cfg_dataset = dataset
        self.cfg_loader = loader
        self.dataset = None

    def prepare_data(self) -> None:
        # Dataset constructors own downloading and processing. Lightning calls this
        # hook on one process, preventing multiple distributed workers from racing.
        if self.dataset is None:
            hydra.utils.instantiate(self.cfg_dataset)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = hydra.utils.instantiate(self.cfg_dataset)

    def _instantiate_loader(self, shuffle: bool) -> NeighborLoader | DataLoader:
        assert self.dataset is not None
        loader = hydra.utils.instantiate(
            self.cfg_loader,
            dataset=self.dataset,
            shuffle=shuffle,
        )
        return loader

    def train_dataloader(self) -> NeighborLoader | DataLoader:
        return self._instantiate_loader(shuffle=True)

    def val_dataloader(self) -> NeighborLoader | DataLoader:
        return self._instantiate_loader(shuffle=False)

    def test_dataloader(self) -> NeighborLoader | DataLoader:
        return self._instantiate_loader(shuffle=False)
