import pytorch_lightning as pl
from models import get_detector_by_name
from metrics import get_metric
import torch
from typing import List,Dict

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
        return self.model.validation_step(batch,batch_idx)

    def validation_epoch_end(self, val_outputs:List) -> Dict:
        for val_output in val_outputs:
            for metric in self.metrics:
                metric(val_output['preds'], val_output['gts'])

        for metric in self.metrics:
            print(str(metric), metric.compute())

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @classmethod
    def build(cls, model_name:str, metric_names:List[str]=[],
            *args, **kwargs) -> pl.LightningModule:
        model = get_detector_by_name(model_name,*args,**kwargs)
        metrics = [get_metric(metric_name) for metric_name in metric_names]
        return cls(model, metrics=metrics)