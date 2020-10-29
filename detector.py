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
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def build_detector(model_name:str) -> pl.LightningModule:
    model = get_detector_by_name(model_name)
    return LightFaceDetector(model)