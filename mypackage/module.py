import pytorch_lightning as pl
import torch
from typing import List,Dict,Any
import numpy as np
import registry

from archs import (
    get_arch_by_name,
    get_arch_config_by_name
)

class FaceDetector(pl.LightningModule):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.__metrics = {}

    def add_metric(self, metric:Dict[str,Any]):
        # TODO fix Any
        self.__metrics.update(metric)

    def forward(self, data:torch.Tensor):
        return self.arch(data)

    def predict(self, data:torch.Tensor, *args, **kwargs):
        return self.arch.predict(data, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.arch.training_step(batch,batch_idx)

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.arch.validation_step(batch,batch_idx)
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
        step_outputs = self.arch.test_step(batch,batch_idx)
        preds = step_outputs.pop('preds',[])
        gts = step_outputs.pop('gts',[])
        for metric in self.__metrics.values():
            metric(preds,gts)
        return step_outputs

    def test_epoch_end(self, test_outputs:List):
        for key,metric in self.__metrics.items():
            self.log(key, metric.compute())

    def configure_optimizers(self):
        return self.arch.configure_optimizers()

    @classmethod
    def build(cls, arch:str, config:Union[str,Dict],
            *args, **kwargs) -> pl.LightningModule:

        # get architecture nn.Module with given configuration
        arch = get_arch_by_name(arch, *args, config=config, **kwargs)

        # build pl.LightninModule with given architecture
        return cls(arch, metrics=metrics)

    @classmethod
    def from_pretrained(cls, model:str, *args, **kwargs) -> pl.LightningModule:
        # TODO validate model

        # TODO parse model

        # TODO call arch.from_pretrained()

        # TODO build pl.LightningModule