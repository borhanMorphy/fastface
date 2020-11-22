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

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch,batch_idx)

    def on_validation_epoch_start(self):
        for metric in self.metrics:
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.model.validation_step(batch,batch_idx)
        preds = step_outputs['preds']
        gts = step_outputs['gts']
        imgs,gt_boxes = batch
        for gt,pred in zip(gts,preds):
            print(gt.long())
            pred[:,:4] = pred[:,:4].long().float()
            print(pred[:5,:])

        return step_outputs

    def validation_epoch_end(self, val_outputs:List):
        print("\nlen val_outputs: ",len(val_outputs))
        preds = []
        gts = []
        losses = []
        for output in val_outputs:
            preds += output['preds']
            gts += output['gts']
            losses.append(output['loss'])
        ap_score = calculate_AP(preds, gts)
        print(f"loss: {sum(losses)/len(losses)} ,AP=0.5 score: {ap_score*100}")

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