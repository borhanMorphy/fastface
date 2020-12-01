import pytorch_lightning as pl
from models import get_detector_by_name
from metrics import get_metric
import torch
from typing import List,Dict
from utils.metric import calculate_AP

import numpy as np
from cv2 import cv2

class LightFaceDetector(pl.LightningModule):
    def __init__(self, model, metrics:List=[]):
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
        for metric in self.metrics:
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.model.validation_step(batch,batch_idx)
        return step_outputs

    def validation_epoch_end(self, val_outputs:List):
        preds = []
        gts = []
        losses = []
        reg_losses = []
        cls_losses = []
        for output in val_outputs:
            preds += output['preds']
            gts += output['gts']
            losses.append(output['loss'])
            cls_losses.append(output['cls_loss'])
            reg_losses.append(output['reg_loss'])
        ap_score = calculate_AP(preds, gts)
        loss = sum(losses)/len(losses)
        cls_loss = sum(cls_losses)/len(cls_losses)
        reg_loss = sum(reg_losses)/len(reg_losses)
        ap_score = (ap_score*100).item()
        print(f"loss: {loss} | cls loss: {cls_loss:.3f} | reg loss: {reg_loss:.3f} | AP=0.5 score: {ap_score:.2f}")
        self.log('val_loss', loss)
        self.log('val_ap', ap_score)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch,batch_idx)

    def test_epoch_end(self, test_outputs:List):
        return self.validation_epoch_end(test_outputs)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @classmethod
    def build(cls, model_name:str, metric_names:List[str]=[],
            *args, **kwargs) -> pl.LightningModule:
        model = get_detector_by_name(model_name,*args,**kwargs)
        metrics = [get_metric(metric_name) for metric_name in metric_names]
        return cls(model, metrics=metrics)

    @classmethod
    def from_pretrained(cls, model_name:str, model_path:str,
            *args,**kwargs):
        model = get_detector_by_name(model_name, *args, **kwargs)
        st = torch.load(model_path, map_location='cpu')
        pl_model = cls(model)
        pl_model.load_state_dict(st['state_dict'])
        return pl_model