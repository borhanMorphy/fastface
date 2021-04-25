import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging

from ..dataset import FDDBDataset
from ..utils import cache
from ..utils.data import default_collate_fn

logger = logging.getLogger("fastface.datamodule")

class FDDBDataModule(pl.LightningDataModule):
    """FDDB pytorch_lightning.LightningDataModule Instance"""

    __URLS__ = {
        # TODO
    }

    def __init__(self, source_dir: str = None,
            batch_size: int = 1, pin_memory: bool = False,
            num_workers: int = 0, collate_fn=default_collate_fn,
            train_transforms=None, val_transforms=None, test_transforms=None):

        super().__init__()
        if source_dir is None:
            source_dir = cache.get_data_cache_path("fddb")

        self.source_dir = source_dir
        self.dl_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "collate_fn": collate_fn
        }

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

    def prepare_data(self, stage: str = None):
        # ! do not make self.<any> = <any> assignments
        # TODO use stage
        # TODO implement here
        pass

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.train_ds = FDDBDataset(self.source_dir,
                phase="train", transforms=self.train_transforms)

            self.val_ds = FDDBDataset(self.source_dir,
                phase="val", transforms=self.val_transforms)
        elif stage == 'test':
            self.test_ds = FDDBDataset(self.source_dir,
                phase="val", transforms=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, **self.dl_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, **self.dl_kwargs)