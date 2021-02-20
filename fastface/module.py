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
    """Generic pl.LightningModule definition for face detection
    """

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
        self.preprocess = transforms

    def add_metric(self, name:str, metric:pl.metrics.Metric):
        """Adds given metric with name key

        Args:
            name (str): name of the metric
            metric (pl.metrics.Metric): Metric object
        """
        # TODO add warnings if override happens
        self.__metrics[name] = metric

    def get_metrics(self) -> Dict[str, pl.metrics.Metric]:
        """Return metrics defined in the `FaceDetector` instance

        Returns:
            Dict[str, pl.metrics.Metric]: defined model metrics with names
        """
        return {k:v for k,v in self.__metrics.items()}

    def forward(self, *args, **kwargs):
        return self.arch(*args, **kwargs)

    @torch.no_grad()
    def predict(self, images:Union[np.ndarray,List], *args, **kwargs) -> Union[Dict,List]:
        """Performs face detection using given image or images

        Args:
            images (Union[np.ndarray,List]): numpy RGB image or list of RGB images

        Returns:
            Union[Dict, List]: prediction result as dictionary. If list of images are given, output also will be list of dictionaries.

        >>> import fastface as ff
        >>> import imageio
        >>> model = ff.FaceDetector.from_pretrained('original_lffd_560_25L_8S').eval()
        >>> img = imageio.imread('resources/friends.jpg')[:,:,:3]
        >>> model.predict(img)
        [{'box': [1049, 178, 1187, 359], 'score': 0.99633336}, {'box': [561, 220, 710, 401], 'score': 0.99252045}]

        """
        assert isinstance(images, (np.ndarray,List)),"given image(s) must be np.ndarray or list of np.ndarrays"
        single_input = isinstance(images,np.ndarray)
        if single_input:
            batch = [images]
        else:
            batch = images

        for image in batch: assert isinstance(image,np.ndarray) and len(image.shape) == 3,"unsupported image type"

        # enable tracking to perform postprocess after inference 
        self.preprocess.enable_tracking()
        # reset queue
        self.preprocess.flush()

        # apply transforms
        batch = torch.stack([self.preprocess(image) for image in batch], dim=0).to(self.device)

        preds:List = []

        for pred in self.arch.predict(batch, *args, **kwargs):
            # postprocess to adjust predictions
            pred = self.preprocess.adjust(pred.cpu().numpy())
            # pred np.ndarray(N,5) as x1,y1,x2,y2,score
            payload = [{'box':person[:4].astype(np.int32).tolist(), 'score':person[4]} for person in pred]
            preds.append(payload)

        # reset queue
        self.preprocess.flush()

        # disable tracking
        self.preprocess.disable_tracking()

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
        """Classmethod for creating `fastface.FaceDetector` instance from scratch

        Args:
            arch (str): architecture name
            config (Union[str,Dict]): configuration name or configuration dictionary
            hparams (Dict, optional): hyper parameters for the model. Defaults to {}.

        Returns:
            pl.LightningModule: fastface.FaceDetector instance with random weights initialization
        """

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
        """Classmethod for creating `fastface.FaceDetector` instance with given checkpoint weights

        Args:
            ckpt_path (str): file path of the checkpoint

        Returns:
            pl.LightningModule: fastface.FaceDetector instance with checkpoint weights
        """
        return cls.load_from_checkpoint(ckpt_path, map_location='cpu')

    @classmethod
    def from_pretrained(cls, model:str, target_path:str=None) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance with pretrained weights

        Args:
            model (str): pretrained model name.
            target_path (str, optional): path to check for model weights, if not given it will use cache path. Defaults to None.

        Returns:
            pl.LightningModule: fastface.FaceDetector instance with pretrained weights
        """
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
        self.preprocess = arch_cls._transforms