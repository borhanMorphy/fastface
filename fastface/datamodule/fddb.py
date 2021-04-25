import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import logging
import os
from typing import List

from ..dataset import FDDBDataset
from ..utils import cache
from ..adapter import download_object
from ..transforms import (
    Compose,
    Interpolate,
    Padding,
    FaceDiscarder,
    Normalize,
    LFFDRandomSample,
    RandomHorizontalFlip
)

logger = logging.getLogger("fastface.datamodule")

class FDDBDataModule(pl.LightningDataModule):
    """FDDB pytorch_lightning.LightningDataModule Instance"""

    __URLS__ = {
        # TODO
    }

    def __init__(self, source_dir:str=None, folds:List[int]=None, **kwargs):
        super().__init__()
        if source_dir is None:
            source_dir = cache.get_data_cache_path("fddb")
        self.source_dir = source_dir
        self.folds = folds

        def default_collate_fn(data):
            imgs,gt_boxes = zip(*data)
            batch = torch.stack(imgs, dim=0)
            return batch,gt_boxes

        default_train_kwargs = {
            'batch_size': 32,
            'pin_memory': True,
            'shuffle': True,
            'num_workers': 8,
            'collate_fn': default_collate_fn
        }
        default_train_kwargs.update(kwargs.get('train_kwargs', {}))
        self.train_kwargs = default_train_kwargs
        default_val_kwargs = {
            'batch_size': 32,
            'pin_memory': True,
            'shuffle': False,
            'num_workers': 8,
            'collate_fn': default_collate_fn
        }
        default_val_kwargs.update(kwargs.get('val_kwargs',{}))
        self.val_kwargs = default_val_kwargs

        default_test_kwargs = {
            'batch_size': 32,
            'pin_memory': True,
            'shuffle': False,
            'num_workers': 8,
            'collate_fn': default_collate_fn
        }
        default_test_kwargs.update(kwargs.get('test_kwargs',{}))
        self.test_kwargs = default_test_kwargs

        self.train_transforms = kwargs.get('train_transforms', Compose(
                Interpolate(max_dim=640),
                Padding(target_size=(640,640), pad_value=0),
                Normalize(mean=0, std=255),
            )
        )

        self.val_transforms = kwargs.get('val_transforms', Compose(
                Interpolate(max_dim=640),
                Padding(target_size=(640,640), pad_value=0),
                Normalize(mean=0, std=255),
            )
        )

        self.test_transforms = kwargs.get('test_transforms', Compose(
                Normalize(mean=0, std=255),
            )
        )

    def prepare_data(self, stage:str=None):
        # ! do not make self.<any> = <any> assignments
        # TODO use stage
        # TODO implement here
        pass

    def setup(self, stage:str=None):
        if stage == 'fit':
            self.train_ds = FDDBDataset(self.source_dir,
                phase="train", transforms=self.train_transforms)

            self.val_ds = FDDBDataset(self.source_dir,
                phase="val", folds=self.folds, transforms=self.val_transforms)
        elif stage == 'test':
            self.test_ds = FDDBDataset(self.source_dir,
                phase="val", folds=self.folds, transforms=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, **self.train_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, **self.val_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, **self.test_kwargs)