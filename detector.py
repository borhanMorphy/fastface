import pytorch_lightning as pl
import torch
from typing import List,Dict
import numpy as np
import registry

from archs import (
    get_arch_by_name,
    get_arch_config_by_name
)

class LightFaceDetector(pl.LightningModule):
    def __init__(self, model, metrics:Dict={}):
        super().__init__()
        self.model = model
        self.metrics = metrics

    def forward(self, data:torch.Tensor):
        return self.model(data)

    def predict(self, data:torch.Tensor, *args, **kwargs):
        return self.model.predict(data, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        losses = self.model.training_step(batch,batch_idx)
        for k,v in losses.items():
            if k != 'loss':
                self.log(k,v,on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return losses['loss']

    def on_validation_epoch_start(self):
        for metric in self.metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.model.validation_step(batch,batch_idx)
        preds = step_outputs.pop('preds',[])
        gts = step_outputs.pop('gts',[])
        for metric in self.metrics.values():
            metric(preds,gts)

        return step_outputs

    def validation_epoch_end(self, val_outputs:List):
        losses = []
        reg_losses = []
        cls_losses = []
        for output in val_outputs:
            losses.append(output['loss'])
            cls_losses.append(output['cls_loss'])
            reg_losses.append(output['reg_loss'])
        loss = sum(losses)/len(losses)
        cls_loss = sum(cls_losses)/len(cls_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        for key,metric in self.metrics.items():
            self.log(key, metric.compute())
        self.log('val_loss', loss)
        self.log('val_cls_loss',cls_loss)
        self.log('val_reg_loss',reg_loss)

    def test_step(self, batch, batch_idx):
        step_outputs = self.model.test_step(batch,batch_idx)
        preds = step_outputs.pop('preds',[])
        gts = step_outputs.pop('gts',[])
        for metric in self.metrics.values():
            metric(preds,gts)
        return step_outputs

    def test_epoch_end(self, test_outputs:List):
        for key,metric in self.metrics.items():
            self.log(key, metric.compute())

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @classmethod
    def build(cls, arch_name:str, config:str='', metrics:Dict={},
            *args, **kwargs) -> pl.LightningModule:
        # get architecture configuration
        arch_config = get_arch_config_by_name(arch_name, config=config)

        # get architecture nn.Module with given configuration
        arch = get_arch_by_name(arch_name, *args, config=arch_config, **kwargs)

        # build pl.LightninModule with given architecture
        return cls(arch, metrics=metrics)

    @classmethod
    def from_pretrained(cls, arch_name:str, weights:str,
            *args, config:str='', metrics:Dict={}, **kwargs) -> pl.LightningDataModule:

        # build pl module
        pl_module = cls.build(arch_name, *args,
            config=config, metrics=metrics, **kwargs)

        # checking if model downloadable
        registry.handle_model_download(weights, arch_name, config)

        # load state dict
        st = torch.load(weights, map_location='cpu')

        # initialize state dict
        if weights.endswith(".ckpt"):
            pl_module.load_state_dict(st['state_dict'])
        else:
            pl_module.load_state_dict(st)
        return pl_module