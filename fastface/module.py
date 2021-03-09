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
from typing import List, Dict, Any, Union
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
from .utils.preprocess import AdaptivePreprocess
import torchvision.ops.boxes as box_ops

class FaceDetector(pl.LightningModule):
    """Generic pl.LightningModule definition for face detection
    """

    def __init__(self, arch: nn.Module = None, preprocess: nn.Module = None, hparams: Dict = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.arch = arch
        self.__metrics = {}
        self.preprocess = preprocess

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

    @torch.no_grad()
    def forward(self, batch: List[torch.Tensor],
            iou_threshold: float=0.4, det_threshold: float=0.4) -> List[torch.Tensor]:
        """list of images with float and C x H x W shape
        * apply preprocess.forward (adaptive batch preprocess or static batch preprocess)
        * call arch.predict
        * apply preprocess.adjust
        * apply filtering using `det_threshold`
        * apply nms (if needed *future)

        Args:
            batch (List[torch.Tensor]): [description]
            iou_threshold (float, optional): [description]. Defaults to 0.4.
            det_threshold (float, optional): [description]. Defaults to 0.4.

        Returns:
            List[torch.Tensor]: preds with shape N x 5 as xmin, ymin, xmax, ymax, score 
        """
        batch, scales, paddings = self.preprocess.forward(batch)
        batch_size = batch.size(0)
        # batch: torch.Tensor(B, C, H, W)
        # scales: torch.Tensor(B,)
        # paddings: torch.Tensor(B, 4)
        
        preds = self.arch.predict(batch)
        # preds: torch.Tensor(B, N, 5)

        preds = self.preprocess.adjust(preds, scales, paddings)
        # preds: torch.Tensor(B, N, 5)

        # filter with det_threshold
        pick_b, pick_n = torch.where(preds[:, :, 4] >= det_threshold)

        boxes = preds[pick_b, pick_n, :4]
        scores = preds[pick_b, pick_n, 4]

        # filter with nms
        # TODO handle if model does not require nms
        pick = box_ops.batched_nms(boxes, scores, pick_b, iou_threshold)
        pick_b = pick_b[pick]
        boxes = boxes[pick]
        scores = scores[pick]

        # TODO sort and select top k
        predictions: List[torch.Tensor] = []

        for i in range(batch_size):
            mask = pick_b == i
            # TODO might cause error
            predictions.append(
                torch.cat([boxes[mask], scores[mask].unsqueeze(-1)], dim=-1)
            )

        return predictions

    @torch.jit.unused
    def predict(self, images:Union[np.ndarray, List], iou_threshold: float = 0.4,
            det_threshold: float = 0.4) -> Union[Dict,List]:
        """Performs face detection using given image or images
        * convert to tensor and H x W x C => C x H x W
        * call self.forward
        * convert to json

        Args:
            images (Union[np.ndarray, List]): numpy RGB image or list of RGB images

        Returns:
            Union[Dict, List]: prediction result as dictionary. If list of images are given, output also will be list of dictionaries.

        >>> import fastface as ff
        >>> import imageio
        >>> model = ff.FaceDetector.from_pretrained('lffd_original').eval()
        >>> img = imageio.imread('resources/friends.jpg')[:,:,:3]
        >>> model.predict(img)
        [{'box': [1049, 178, 1187, 359], 'score': 0.99633336}, {'box': [561, 220, 710, 401], 'score': 0.99252045}]

        """
        # convert images to list of tensors
        batch = self.to_tensor(images, dtype=self.dtype, device=self.device)
        # batch: List[torch.Tensor(C, H, W), ...]

        preds = self.forward(batch, iou_threshold=iou_threshold,
            det_threshold=det_threshold)
        # preds: List[torch.Tensor(N, 5), ...]

        results = self.to_json(preds)
        """results
        [
            # single image results
            {
                "boxes": <array>,  # List[List[xmin, ymin, xmax, ymax]]
                "scores": <array>  # List[float]
            },
            ...
        ]
        """
        return results

    @staticmethod
    def to_tensor(images: Union[np.ndarray, List],
            dtype=torch.float32, device: str = 'cpu') -> List[torch.Tensor]:
        """Converts given image or list of images to list of tensors

        Args:
            images (Union[np.ndarray, List]): RGB image or list of RGB images
            dtype (torch.dtype) : data type of the tensor, default: torch.float32
            device (str) : device of the tensor, default: 'cpu' 

        Returns:
            List[torch.Tensor]: list of torch.Tensor(C x H x W)
        """
        if isinstance(images, np.ndarray):
            images = [images]

        batch: List[torch.Tensor] = []

        for img in images:
            # TODO check shape
            batch.append(
                # h,w,c => c,h,w
                torch.tensor(img, dtype=dtype, device=device).permute(2,0,1)
            )

        return batch

    @staticmethod
    def to_json(preds: List[torch.Tensor]) -> List[Dict]:
        """Converts given list of tensor predictions to json serializable format

        Args:
            preds (List[torch.Tensor]): list of torch.Tensor(N,5) as xmin, ymin, xmax, ymax, score

        Returns:
            List[Dict]: [
                # single image results
                {
                    "boxes": <array>,  # List[List[xmin, ymin, xmax, ymax]]
                    "scores": <array>  # List[float]
                },
                ...
            ]
        """
        results: List[Dict] = []

        for pred in preds:
            if pred.size(0) != 0:
                pred = pred.cpu().numpy()
                boxes = pred[:, :4].astype(np.int32).tolist()
                scores = pred[:, 4].tolist()
            else:
                boxes = [],
                scores = []

            results.append({
                "boxes": boxes,
                "scores": scores
            })

        return results

    def training_step(self, batch, batch_idx):
        return self.arch.training_step(batch,batch_idx,**self.hparams)

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
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
            metric_value = metric.compute()
            self.log(key, metric_value)
        self.log('val_loss', loss)

    def on_test_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    @torch.no_grad()
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
    def build(cls, arch: str, config: Union[str, Dict],
            hparams: Dict = {}, mean: List[float] = [0.0, 0.0, 0.0],
            std: List[float] = [255., 255., 255.], **kwargs) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance from scratch

        Args:
            arch (str): architecture name
            config (Union[str,Dict]): configuration name or configuration dictionary
            hparams (Dict, optional): hyper parameters for the model. Defaults to {}.

        Returns:
            pl.LightningModule: fastface.FaceDetector instance with random weights initialization
        """

        # get architecture configuration if needed
        config = config if isinstance(config, Dict) else get_arch_config(arch, config)

        # get architecture nn.Module class
        arch_cls = get_arch_cls(arch)

        # build nn.Module with given configuration
        arch_module = arch_cls(config=config, **kwargs)

        # build preprocess nn.Module
        preprocess = AdaptivePreprocess(mean, std)

        # add config and arch information to the hparams
        hparams.update({'config': config, 'arch': arch})

        # add kwargs to the hparams
        hparams.update({'kwargs': kwargs})

        # build pl.LightninModule with given architecture
        return cls(arch=arch_module, preprocess=preprocess, hparams=hparams)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance with given checkpoint weights

        Args:
            ckpt_path (str): file path of the checkpoint

        Returns:
            pl.LightningModule: fastface.FaceDetector instance with checkpoint weights
        """
        return cls.load_from_checkpoint(ckpt_path, map_location='cpu', **kwargs)

    @classmethod
    def from_pretrained(cls, model: str, target_path: str = None, **kwargs) -> pl.LightningModule:
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
        return cls.from_checkpoint(model, **kwargs)

    def on_load_checkpoint(self, checkpoint: Dict):
        arch = checkpoint['hyper_parameters']['arch']
        config = checkpoint['hyper_parameters']['config']
        kwargs = checkpoint['hyper_parameters']['kwargs']

        # get architecture nn.Module class
        arch_cls = get_arch_cls(arch)

        # get architecture configuration if needed
        config = config if isinstance(config, Dict) else get_arch_config(arch, config)

        # build nn.Module with given configuration
        self.arch = arch_cls(config=config, **kwargs)

        # build preprocess nn.Module
        self.preprocess = AdaptivePreprocess([0.0, 0.0, 0.0], [255., 255., 255.])