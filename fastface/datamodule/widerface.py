import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import logging
import os
from typing import List

from ..dataset import WiderFaceDataset
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

class WiderFaceDataModule(pl.LightningDataModule):
    """Widerface pytorch_lightning.LightningDataModule Instance"""

    __URLS__ = {
        'widerface-train': {
            'adapter': 'gdrive',
            'check':'WIDER_train',
            'kwargs':{
                'file_id': '0B6eKvaijfFUDQUUwd21EckhUbWs',
                'file_name': 'WIDER_train.zip',
                'unzip': True
            }
        },
        'widerface-val': {
            'adapter': 'gdrive',
            'check':'WIDER_val',
            'kwargs':{
                'file_id':'0B6eKvaijfFUDd3dIRmpvSk8tLUk',
                'file_name': 'WIDER_val.zip',
                'unzip': True
            }
        },
        'widerface-annotations': {
            'adapter': 'http',
            'check':'wider_face_split',
            'kwargs':{
                'url': 'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip',
                'unzip': True
            }
        },
        'widerface-eval-code': {
            'adapter': 'http',
            'check':'eval_tools',
            'kwargs':{
                'url': 'http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip',
                'unzip': True
            }
        }
    }

    def __init__(self, source_dir:str=None, partitions:List[str]=None, **kwargs):
        super().__init__()
        if source_dir is None:
            source_dir = cache.get_data_cache_path("widerface")
        self.source_dir = source_dir
        self.partitions = partitions

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

        self.train_transform = kwargs.get('train_transform')
        self.train_target_transform = kwargs.get('train_target_transform')
        self.train_transforms = kwargs.get('train_transforms', Compose(
                FaceDiscarder(min_face_size=2),
                LFFDRandomSample( # TODO handle different configurations
                    [
                        (10,15),(15,20),(20,40),(40,70),
                        (70,110),(110,250),(250,400),(400,560)
                    ], target_size=(640,640)),
                FaceDiscarder(min_face_size=8),
                RandomHorizontalFlip(p=0.5),
                Normalize(mean=127.5, std=127.5)
            )
        )

        self.val_transform = kwargs.get('val_transform')
        self.val_target_transform = kwargs.get('val_target_transform')
        self.val_transforms = kwargs.get('val_transforms', Compose(
                Interpolate(max_dim=640),
                Padding(target_size=(640,640), pad_value=0),
                Normalize(mean=127.5, std=127.5)
            )
        )

        self.test_transform = kwargs.get('test_transform')
        self.test_target_transform = kwargs.get('test_target_transform')
        self.test_transforms = kwargs.get('test_transforms', Compose(
                Normalize(mean=127.5, std=127.5)
            )
        )

    def prepare_data(self, stage:str=None):
        # ! do not make self.<any> = <any> assignments
        # TODO use stage
        logger.info(f"checking folders for widerface({stage}) datamodule...")
        for k,v in self.__URLS__.items():
            if stage == 'test' and k == 'widerface-train':
                # skip downloading train data when stage is `test`
                continue

            logger.debug(f"checking {k}")
            check = v.get('check')
            check_path = os.path.join(self.source_dir,check)
            if os.path.exists(check_path):
                logger.debug(f"found {k} at {check_path}")
                continue
            # download
            adapter = v.get('adapter')
            kwargs = v.get('kwargs', {})
            logger.warning(f"downloading via adapter:{adapter} to: {check_path} with: {kwargs}")
            download_object(adapter, dest_path=self.source_dir, **kwargs)

    def setup(self, stage:str=None):
        if stage == 'fit':
            self.train_ds = WiderFaceDataset(self.source_dir,
                phase="train", transforms=self.train_transforms)

            self.val_ds = WiderFaceDataset(self.source_dir,
                phase="val", partitions=self.partitions,
                transforms=self.val_transforms)
        elif stage == 'test':
            self.test_ds = WiderFaceDataset(self.source_dir,
                phase="val", partitions=self.partitions,
                transforms=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        # TODO fix this
        return DataLoader(self.val_ds, **self.val_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, **self.val_kwargs)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, **self.test_kwargs)