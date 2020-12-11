import pytorch_lightning as pl
from models import get_detector_by_name
import torch
from typing import List,Dict

import numpy as np

class LightFaceDetector(pl.LightningModule):
    def __init__(self, model, metrics:Dict={}):
        super().__init__()
        self.model = model
        self.metrics = metrics

    def forward(self, data:torch.Tensor):
        return self.model(data)

    def predict(self, data:torch.Tensor):
        return self.model.predict(data)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch,batch_idx)

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
    def build(cls, model_name:str, metrics:Dict=[],
            *args, **kwargs) -> pl.LightningModule:
        model = get_detector_by_name(model_name,*args,**kwargs)
        return cls(model, metrics=metrics)

    @classmethod
    def from_pretrained(cls, model_name:str, model_path:str,
            *args,**kwargs):
        metrics = kwargs.pop('metrics')

        model = get_detector_by_name(model_name, *args, **kwargs)
        st = torch.load(model_path, map_location='cpu')
        pl_model = cls(model, metrics=metrics)
        if model_path.endswith(".ckpt"):
            pl_model.load_state_dict(st['state_dict'])
        else:
            pl_model.load_state_dict(st)
        return pl_model