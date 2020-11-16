import pytorch_lightning as pl
from models import get_detector_by_name
import torch

class LightFaceDetector(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data:torch.Tensor):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch,batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch,batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @classmethod
    def build(cls, model_name:str, *args, **kwargs) -> pl.LightningModule:
        model = get_detector_by_name(model_name,*args,**kwargs)
        return cls(model)