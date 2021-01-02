import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import List,Dict,Any,Union
import numpy as np
import os

from .api import (
    get_arch_config,
    download_pretrained_model
)

from .utils.config import (
    get_arch_cls
)

from .utils.cache import get_model_cache_path

class FaceDetector(pl.LightningModule):
    def __init__(self, arch:nn.Module=None, hparams:Dict=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.arch = arch
        self.__metrics = {}

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        self.__metrics[name] = metric

    def forward(self, data:torch.Tensor):
        return self.arch(data)

    def predict(self, data:torch.Tensor, *args, **kwargs):
        return self.arch.predict(data.to(self.device), *args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.arch.training_step(batch,batch_idx,**self.hparams)

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.arch.validation_step(batch,batch_idx,**self.hparams)
        preds = step_outputs.pop('preds',[])
        gts = step_outputs.pop('gts',[])
        for metric in self.__metrics.values():
            metric(preds,gts)

        return step_outputs

    def validation_epoch_end(self, val_outputs:List):
        losses = []
        for output in val_outputs:
            losses.append(output['loss'])
        loss = sum(losses)/len(losses)
        for key,metric in self.__metrics.items():
            self.log(key, metric.compute())
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        step_outputs = self.arch.test_step(batch,batch_idx,**self.hparams)
        preds = step_outputs.pop('preds',[])
        gts = step_outputs.pop('gts',[])
        for metric in self.__metrics.values():
            metric(preds,gts)
        return step_outputs

    def test_epoch_end(self, test_outputs:List):
        for key,metric in self.__metrics.items():
            self.log(key, metric.compute())

    def configure_optimizers(self):
        return self.arch.configure_optimizers(**self.hparams)

    @classmethod
    def build(cls, arch:str, config:Union[str,Dict], hparams:Dict={}, **kwargs) -> pl.LightningModule:

        # get architecture configuration if needed
        config = config if isinstance(config,Dict) else get_arch_config(arch,config)

        # get architecture nn.Module class
        arch_cls = get_arch_cls(arch)

        # build nn.Module with given configuration
        arch_module = arch_cls(config=config, **kwargs)

        # add config and arch information to the hparams
        hparams.update({'config':config,'arch':arch})

        # add kwargs to the hparams
        hparams.update({'kwargs':kwargs})

        # build pl.LightninModule with given architecture
        return cls(arch=arch_module, hparams=hparams)

    @classmethod
    def from_checkpoint(cls, ckpt_path:str) -> pl.LightningModule:
        # build pl.LightninModule from checkpoint
        return cls.load_from_checkpoint(ckpt_path, map_location='cpu')

    @classmethod
    def from_pretrained(cls, model:str=None, model_path:str=None,
            target_path:str=None) -> pl.LightningModule:

        if model_path is None:
            assert model is not None,"model cannot be `None` if model path is `None`"
            model_path = download_pretrained_model(model, target_path=target_path)
        else:
            assert os.path.isfile(model_path),f"model path is given but not found in the disk: {model_path}"
        return cls.load_from_checkpoint(model_path, map_location='cpu')        

    def on_load_checkpoint(self, checkpoint:Dict):
        arch = checkpoint['hyper_parameters']['arch']
        config = checkpoint['hyper_parameters']['config']
        kwargs = checkpoint['hyper_parameters']['kwargs']

        # get architecture nn.Module class
        arch_cls = get_arch_cls(arch)

        # get architecture configuration if needed
        config = config if isinstance(config,Dict) else get_arch_config(arch,config)

        # build nn.Module with given configuration
        self.arch  = arch_cls(config=config, **kwargs)

        