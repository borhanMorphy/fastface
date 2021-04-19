import os
from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .api import (
    get_arch_config,
    download_pretrained_model,
    list_pretrained_models
)

from .utils.config import (
    get_arch_cls
)

from .utils.cache import get_model_cache_path
from .utils.preprocess import prepare_batch, adjust_results
from .utils import box as box_ops

class FaceDetector(pl.LightningModule):
    """Generic pl.LightningModule definition for face detection
    """

    def __init__(self, arch: nn.Module = None, mean: List[float] = [0.0, 0.0, 0.0],
            std: List[float] = [255., 255., 255.], hparams: Dict = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.arch = arch
        self.__metrics = {}
        # pylint: disable=not-callable
        self.register_buffer("mean", torch.tensor(
            mean, dtype=self.dtype, device=self.device).reshape(1, len(mean), 1, 1))
        # pylint: disable=not-callable
        self.register_buffer("std", torch.tensor(
            std, dtype=self.dtype, device=self.device).reshape(1, len(std), 1, 1))

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
    def forward(self, batch: torch.Tensor,
            iou_threshold: float = 0.4, det_threshold: float = 0.6) -> Tuple[torch.Tensor, torch.Tensor]:
        """list of images with float and C x H x W shape

        Args:
            batch (torch.Tensor): torch.FloatTensor(B x C x H x W)
            iou_threshold (float, optional): iou threshold. Defaults to 0.4.
            det_threshold (float, optional): detection threshold. Defaults to 0.6.

        Returns:
            Tuple[torch.Tensor]:
                (0): contains batch ids for predictions as N,
                (1): preds with shape N x 5 as batch_idx, xmin, ymin, xmax, ymax, score
        """
        batch_size = batch.shape[0]

        # apply normalization
        batch = (batch - self.mean) / self.std

        preds = self.arch.predict(batch)
        # preds: torch.Tensor(B, N, 5)

        # filter with det_threshold
        pick_b, pick_n = torch.where(preds[:, :, 4] >= det_threshold)

        boxes = preds[pick_b, pick_n, :4]
        scores = preds[pick_b, pick_n, 4]

        # filter with nms
        # TODO handle if model does not require nms
        # TODO make use of top_k

        preds,batch_ids = box_ops.batched_nms(boxes, scores, pick_b, iou_threshold=iou_threshold)

        return batch_ids, preds

    @torch.jit.unused
    def predict(self, images:Union[np.ndarray, List], iou_threshold: float = 0.4,
            det_threshold: float = 0.4, adaptive_batch: bool = True) -> List:
        """Performs face detection using given image or images

        Args:
            images (Union[np.ndarray, List]): numpy RGB image or list of RGB images
            iou_threshold (float): iou value threshold for nms, Default: 0.4
            det_threshold (float): detection score threshold, Default: 0.4
            adaptive_batch (bool): if true than batching will be adaptive,
                using max dimension of the batch, otherwise it will use static image size, Default: True

        Returns:
            List: prediction result as list of dictionaries.
            [
                # single image results
                {
                    "boxes": <array>,  # List[List[xmin, ymin, xmax, ymax]]
                    "scores": <array>  # List[float]
                },
                ...
            ]
        >>> import fastface as ff
        >>> import imageio
        >>> model = ff.FaceDetector.from_pretrained('lffd_original').eval()
        >>> img = imageio.imread('resources/friends.jpg')[:,:,:3]
        >>> model.predict(img)
        [{'boxes': [[1055, 177, 1187, 356], [574, 225, 704, 391], [129, 217, 263, 381], [321, 231, 447, 390], [858, 265, 977, 410]], 'scores': [0.9999922513961792, 0.9999910593032837, 0.9999610185623169, 0.9999467134475708, 0.9874875545501709]}]
        """
        # convert images(uint8) to list of tensors(float)
        batch = self.to_tensor(images)
        # batch: List[torch.Tensor(C, H, W), ...]

        batch_size = len(batch)

        # prepare batch
        batch, scales, paddings = prepare_batch(batch, self.arch.input_shape[-1],
            adaptive_batch=adaptive_batch)

        batch = batch.to(self.device, self.dtype)

        # batch: torch.Tensor(C,)
        batch_ids, preds = self.forward(batch, iou_threshold=iou_threshold,
            det_threshold=det_threshold)

        preds = preds.cpu()
        batch_ids = batch_ids.cpu()
        # preds: torch.Tensor(N, 5)
        # batch_ids: torch.Tensor(N,)

        # convert to list
        preds: List[torch.Tensor] = [preds[batch_ids == i] for i in range(batch_size)]

        # adjust results
        preds = adjust_results(preds, scales, paddings)

        # convert predictions to json serializeable format
        results = self.to_json(preds)

        return results

    @staticmethod
    def to_tensor(images: Union[np.ndarray, List]) -> List[torch.Tensor]:
        """Converts given image or list of images to list of tensors

        Args:
            images (Union[np.ndarray, List]): RGB image or list of RGB images

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
                # pylint: disable=not-callable
                torch.tensor(img, dtype=torch.float32).permute(2,0,1)
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
            print(key, metric_value)
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
            std: List[float] = [255.0, 255.0, 255.0], **kwargs) -> pl.LightningModule:
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

        # add config and arch information to the hparams
        hparams.update({'config': config, 'arch': arch})

        # add kwargs to the hparams
        hparams.update({'kwargs': kwargs})

        # build pl.LightninModule with given architecture
        return cls(arch=arch_module, mean=mean, std=std, hparams=hparams)

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