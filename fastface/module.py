from .transform import (
    Compose,
    Interpolate,
    Padding,
    Normalize,
    ToTensor
)

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import List,Dict,Any,Union
import numpy as np
import os

from .api import (
    get_arch_config,
    download_pretrained_model,
    list_pretrained_models
)

from .utils.config import (
    get_arch_cls
)

from .utils.cache import get_model_cache_path

class FaceDetector(pl.LightningModule):
    def __init__(self, arch:nn.Module=None, transforms:Compose=None, hparams:Dict=None):
        super().__init__()
        if isinstance(transforms,type(None)):
            transforms = Compose(
                Interpolate(max_dim=640),
                Padding(target_size=(640,640)),
                Normalize(mean=127.5, std=127.5),
                ToTensor()
            )
        assert isinstance(transforms, Compose),f"given transforms must be instance of Compose, but found:{type(transforms)}"
        self.save_hyperparameters(hparams)
        self.arch = arch
        self.__metrics = {}
        self._transforms = transforms

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        self.__metrics[name] = metric

    def get_metrics(self) -> Dict[str, pl.metrics.Metric]:
        return {k:v for k,v in self.__metrics.items()}

    def forward(self, *args, **kwargs):
        return self.arch(*args, **kwargs)

    @torch.no_grad()
    def predict(self, images:Union[np.ndarray,List], *args, **kwargs) -> Union[Dict,List]:
        assert isinstance(images, (np.ndarray,List)),"given image(s) must be np.ndarray or list of np.ndarrays"
        single_input = isinstance(images,np.ndarray)
        if single_input:
            batch = [images]
        else:
            batch = images

        for image in batch: assert isinstance(image,np.ndarray) and len(image.shape) == 3,"unsupported image type"

        # enable tracking to perform postprocess after inference 
        self._transforms.enable_tracking()
        # reset queue
        self._transforms.flush()

        # apply transforms
        batch = torch.stack([self._transforms(image) for image in batch], dim=0).to(self.device)

        preds:List = []

        for pred in self.arch.predict(batch, *args, **kwargs):
            # postprocess to adjust predictions
            pred = self._transforms.adjust(pred.cpu().numpy())
            # pred np.ndarray(N,5) as x1,y1,x2,y2,score
            payload = [{'box':person[:4].astype(np.int32).tolist(), 'score':person[4]} for person in pred]
            preds.append(payload)

        # reset queue
        self._transforms.flush()

        # disable tracking
        self._transforms.disable_tracking()

        return preds[0] if single_input else preds

    def training_step(self, batch, batch_idx):
        return self.arch.training_step(batch,batch_idx,**self.hparams)

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        step_outputs = self.arch.validation_step(batch,batch_idx,**self.hparams)
        if "preds" in step_outputs and "gts" in step_outputs:
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

    def on_test_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def test_step(self, batch, batch_idx):
        step_outputs = self.arch.test_step(batch,batch_idx,**self.hparams)
        if "preds" in step_outputs and "gts" in step_outputs:
            preds = step_outputs.pop('preds',[])
            gts = step_outputs.pop('gts',[])
            for metric in self.__metrics.values():
                metric(preds,gts)
        return step_outputs

    def test_epoch_end(self, test_outputs:List):
        for key,metric in self.__metrics.items():
            self.log(key, metric.compute())
        losses = []
        for test_output in test_outputs:
            if 'loss' not in test_output: continue
            losses.append(test_output['loss'])

        if len(losses) != 0:
            loss = sum(losses) / len(losses)
            self.log("test_loss: ", loss)

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
    def from_pretrained(cls, model:str, target_path:str=None) -> pl.LightningModule:
        if model in list_pretrained_models():
            model = download_pretrained_model(model, target_path=target_path)
        assert os.path.isfile(model),f"given {model} not found in the disk"
        return cls.from_checkpoint(model)

    def on_load_checkpoint(self, checkpoint:Dict):
        arch = checkpoint['hyper_parameters']['arch']
        config = checkpoint['hyper_parameters']['config']
        kwargs = checkpoint['hyper_parameters']['kwargs']

        # get architecture nn.Module class
        arch_cls = get_arch_cls(arch)

        # get architecture configuration if needed
        config = config if isinstance(config,Dict) else get_arch_config(arch,config)

        # build nn.Module with given configuration
        self.arch = arch_cls(config=config, **kwargs)
        self._transforms = arch_cls._transforms