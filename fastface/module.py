import math
import os
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from torchmetrics.metric import Metric

from . import api, utils
from .config import ArchConfig


class FaceDetector(pl.LightningModule):
    """Generic pl.LightningModule definition for face detection"""

    def __init__(
        self,
        arch: nn.Module = None,
        normalized_input: bool = False,
        mean: Union[float, List] = 0.0,
        std: Union[float, List] = 1.0,
        hparams: Dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.arch = arch
        self.__metrics = dict()
        self.every_n_epoch = math.inf
        self.every_n_batch = math.inf

        self.init_preprocess(mean=mean, std=std, normalized_input=normalized_input)

    def is_debug_step(self, batch_idx: int) -> bool:
        return (self.current_epoch + 1) % self.every_n_epoch == 0 and (
            batch_idx + 1
        ) % self.every_n_batch == 0

    def init_preprocess(
        self,
        normalized_input: bool = False,
        mean: Union[float, List] = 0.0,
        std: Union[float, List] = 1.0,
    ):

        # preprocess
        if isinstance(mean, list):
            assert len(mean) == 3, "mean dimension must be 3 not {}".format(len(mean))
            mean = [float(m) for m in mean]
        else:
            mean = [float(mean) for _ in range(3)]

        if isinstance(std, list):
            assert len(std) == 3, "std dimension must be 3 not {}".format(len(std))
            std = [float(m) for m in std]
        else:
            std = [float(std) for _ in range(3)]

        self.register_buffer(
            "normalizer",
            torch.tensor(255.0)
            if normalized_input
            else torch.tensor(1.0),  # pylint: disable=not-callable
            persistent=False,
        )

        self.register_buffer(
            "mean",
            torch.tensor(mean)
            .view(-1, 1, 1)
            .contiguous(),  # pylint: disable=not-callable
            persistent=False,
        )

        self.register_buffer(
            "std",
            torch.tensor(std)
            .view(-1, 1, 1)
            .contiguous(),  # pylint: disable=not-callable
            persistent=False,
        )

    def add_metric(self, name: str, metric: Metric):
        """Adds given metric with name key

        Args:
                name (str): name of the metric
                metric (Metric): Metric object
        """
        # TODO add warnings if override happens
        self.__metrics[name] = metric

    def get_metrics(self) -> Dict[str, Metric]:
        """Return metrics defined in the `FaceDetector` instance

        Returns:
                Dict[str, Metric]: defined model metrics with names
        """
        return {k: v for k, v in self.__metrics.items()}

    @torch.jit.unused
    def predict(
        self,
        data: Union[np.ndarray, List],
        target_size: int = None,
        det_threshold: float = 0.4,
        iou_threshold: float = 0.4,
        keep_n: int = 200,
    ):
        """Performs face detection using given image or images

        Args:
                data (Union[np.ndarray, List]): numpy RGB image or list of RGB images
                target_size (int): if given than images will be up or down sampled using `target_size`, Default: None
                det_threshold (float): detection score threshold, Default: 0.4
                iou_threshold (float): iou value threshold for nms, Default: 0.4
                keep_n (int): describes how many prediction will be selected for each batch, Default: 200

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
        >>> model.predict(img, target_size=640)
        [{'boxes': [[1057, 180, 1187, 352], [571, 212, 697, 393], [140, 218, 270, 382], [864, 271, 979, 406], [327, 252, 442, 392]], 'scores': [0.9992133378982544, 0.9971852898597717, 0.9246336817741394, 0.8549349308013916, 0.8072562217712402]}]
        """

        batch = self.to_tensor(data)
        # batch: list of tensors

        batch_size = len(batch)

        batch, scales, paddings = utils.preprocess.prepare_batch(
            batch, target_size=target_size, adaptive_batch=target_size is None
        )
        # batch: torch.Tensor(B,C,T,T)
        # scales: torch.Tensor(B,)
        # paddings: torch.Tensor(B,4) as pad (left, top, right, bottom)

        preds = self.forward(
            batch,
            det_threshold=det_threshold,
            iou_threshold=iou_threshold,
            keep_n=keep_n,
        )
        # preds: torch.Tensor(B, N, 6) as x1,y1,x2,y2,score,batch_idx

        preds = [preds[preds[:, 5] == batch_idx, :5] for batch_idx in range(batch_size)]
        # preds: list of torch.Tensor(N, 5) as x1,y1,x2,y2,score

        preds = utils.preprocess.adjust_results(preds, scales, paddings)
        # preds: list of torch.Tensor(N, 5) as x1,y1,x2,y2,score

        preds = self.to_json(preds)
        # preds: list of {"boxes": [(x1,y1,x2,y2), ...], "score": [<float>, ...]}

        return preds

    # ! use forward only for inference not training
    def forward(
        self,
        batch: torch.Tensor,
        det_threshold: float = 0.3,
        iou_threshold: float = 0.4,
        keep_n: int = 200,
    ) -> torch.Tensor:
        """batch of images with float and B x C x H x W shape

        Args:
                batch (torch.Tensor): torch.FloatTensor(B x C x H x W)

        Returns:
                torch.Tensor: preds with shape (N, 6);
                        (1:4) xmin, ymin, xmax, ymax
                        (4:5) score
                        (5:6) batch idx
        """

        # apply preprocess
        batch = ((batch / self.normalizer) - self.mean) / self.std

        # get logits
        with torch.no_grad():
            logits = self.arch.forward(batch)
        # logits, any

        preds = self.arch.compute_preds(logits)
        # preds: torch.Tensor(B, N, 5)

        batch_preds = self._postprocess(
            preds,
            det_threshold=det_threshold,
            iou_threshold=iou_threshold,
            keep_n=keep_n,
        )

        return batch_preds

    def _postprocess(
        self,
        preds: torch.Tensor,
        det_threshold: float = 0.3,
        iou_threshold: float = 0.4,
        keep_n: int = 200,
    ) -> torch.Tensor:

        # TODO pydoc
        batch_size = preds.size(0)

        # filter out some predictions using score
        pick_b, pick_n = torch.where(preds[:, :, 4] >= det_threshold)

        # add batch_idx dim to preds
        preds = torch.cat(
            [preds[pick_b, pick_n, :], pick_b.to(preds.dtype).unsqueeze(1)], dim=1
        )
        # preds: N x 6

        batch_preds: List[torch.Tensor] = []
        for batch_idx in range(batch_size):
            (pick_n,) = torch.where(batch_idx == preds[:, 5])
            order = preds[pick_n, 4].sort(descending=True)[1]

            batch_preds.append(
                # preds: n, 6
                preds[pick_n, :][order][:keep_n, :]
            )

        batch_preds = torch.cat(batch_preds, dim=0)
        # batch_preds: N x 6

        # filter with nms
        pick = utils.box.batched_nms(
            batch_preds[:, :4],  # boxes as x1,y1,x2,y2
            batch_preds[:, 4],  # det score between [0, 1]
            batch_preds[:, 5],  # id of the batch that prediction belongs to
            iou_threshold=iou_threshold,
        )

        # select picked preds
        batch_preds = batch_preds[pick, :]
        # batch_preds: N x 6
        return batch_preds

    def _step(self, batch, batch_idx, phase: str = None):
        inputs, raw_targets = batch

        # apply preprocess
        inputs = ((inputs / self.normalizer) - self.mean) / self.std

        if phase == "train":
            # compute logits
            logits = self.arch.forward(inputs)
        else:
            with torch.no_grad():
                # compute logits
                logits = self.arch.forward(inputs)

            batch_size = inputs.shape[0]

            preds = self.arch.compute_preds(logits)
            # preds: B x N x (5 + 2*l)
            # xmin, ymin, xmax, ymax, score, *landmarks

            preds = self._postprocess(preds)
            # preds: N x 6

            pred_boxes = [
                preds[preds[:, 5] == batch_idx][:, :5].cpu()
                for batch_idx in range(batch_size)
            ]
            target_boxes = [target["bboxes"].cpu() for target in raw_targets]
            labels = [target["labels"] for target in raw_targets]

            for metric in self.__metrics.values():
                metric.update(pred_boxes, target_boxes, labels)

        target_logits = self.arch.build_targets(inputs, raw_targets)

        # compute loss
        # loss: dict of losses or loss
        return self.arch.compute_loss(logits, target_logits, hparams=self.hparams["hparams"])

    def _on_epoch_start(self, phase: str = None):
        for metric in self.__metrics.values():
            metric.reset()

    def _epoch_end(self, outputs, phase: str = None):
        losses = dict()
        for output in outputs:
            if isinstance(output, dict):
                for k, v in output.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v)
            else:
                # only contains `loss`
                if "loss" not in losses:
                    losses["loss"] = []
                losses["loss"].append(output)

        for name, loss in losses.items():
            self.log("{}/{}".format(name, phase), sum(loss) / len(loss))

        if phase != "train":
            for name, metric in self.__metrics.items():
                self.log("{}/{}".format(name, phase), metric.compute())

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="train")

    def on_validation_epoch_start(self):
        self._on_epoch_start(phase="validation")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="validation")

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, phase="validation")

    def on_test_epoch_start(self):
        self._on_epoch_start(phase="test")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="test")

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, phase="test")

    def configure_optimizers(self):
        return self.arch.configure_optimizers(hparams=self.hparams["hparams"])

    @classmethod
    def build_from_yaml(cls, yaml_file_path: str) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance from scratch using yaml file

        Args:
                yaml_file_path (str): yaml file path

        Returns:
                pl.LightningModule: fastface.FaceDetector instance with random weights initialization
        """

        assert os.path.isfile(
            yaml_file_path
        ), "could not find the yaml file given {}".format(yaml_file_path)
        with open(yaml_file_path, "r") as foo:
            yaml_config = yaml.load(foo, Loader=yaml.FullLoader)

        assert "arch" in yaml_config, "yaml file must contain `arch` key"
        assert "config" in yaml_config, "yaml file must contain `config` key"

        arch = yaml_config["arch"]
        config = yaml_config["config"]
        preprocess = yaml_config.get(
            "preprocess", {"mean": 0.0, "std": 1.0, "normalized_input": True}
        )
        hparams = yaml_config.get("hparams", {})

        return cls.build(arch, config, **preprocess, hparams=hparams)

    @classmethod
    def build(
        cls,
        arch: str,
        config: Union[str, ArchConfig],
        preprocess: Dict = {"mean": 0.0, "std": 1.0, "normalized_input": True},
        hparams: Dict = {},
        **kwargs,
    ) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance from scratch

        Args:
                arch (str): architecture name
                config (Union[str, Dict]): configuration name or configuration dictionary
                preprocess (Dict, optional): preprocess arguments of the module. Defaults to {"mean": 0, "std": 1, "normalized_input": True}.
                hparams (Dict, optional): hyper parameters for the model. Defaults to {}.

        Returns:
                pl.LightningModule: fastface.FaceDetector instance with random weights initialization
        """
        assert isinstance(preprocess, dict), "preprocess must be dict, not {}".format(
            type(preprocess)
        )

        # build nn.Module with given configuration
        arch_module = api.build_arch(arch, config)

        module_params = {}

        # add hparams
        module_params.update({"hparams": hparams})

        # add preprocess to the hparams
        module_params.update({"preprocess": preprocess})

        # add config and arch information to the hparams
        module_params.update({"config": config, "arch": arch})

        # add kwargs to the hparams
        module_params.update({"kwargs": kwargs})

        # build pl.LightninModule with given architecture
        return cls(arch=arch_module, **preprocess, hparams=module_params)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, **kwargs) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance, using checkpoint file path

        Args:
                ckpt_path (str): file path of the checkpoint

        Returns:
                pl.LightningModule: fastface.FaceDetector instance with checkpoint weights
        """
        return cls.load_from_checkpoint(ckpt_path, map_location="cpu", **kwargs)

    @classmethod
    def from_pretrained(
        cls, model: str, target_path: str = None, **kwargs
    ) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance, using model name

        Args:
                model (str): pretrained model name.
                target_path (str, optional): path to check for model weights, if not given it will use cache path. Defaults to None.

        Returns:
                pl.LightningModule: fastface.FaceDetector instance with pretrained weights
        """
        if model in api.list_pretrained_models():
            model = api.download_pretrained_model(model, target_path=target_path)
        assert os.path.isfile(model), f"given {model} not found in the disk"
        return cls.from_checkpoint(model, **kwargs)

    def on_load_checkpoint(self, checkpoint: Dict):
        arch = checkpoint["hyper_parameters"]["arch"]
        config = checkpoint["hyper_parameters"]["config"]
        preprocess = checkpoint["hyper_parameters"]["preprocess"]

        kwargs = checkpoint["hyper_parameters"]["kwargs"]

        print("kwargs:::", kwargs)

        # build nn.Module with given configuration
        self.arch = api.build_arch(arch, config)

        # initialize preprocess with given arguments
        self.init_preprocess(**preprocess)

    def to_tensor(self, images: Union[np.ndarray, List]) -> List[torch.Tensor]:
        """Converts given image or list of images to list of tensors
        Args:
                images (Union[np.ndarray, List]): RGB image or list of RGB images
        Returns:
                List[torch.Tensor]: list of torch.Tensor(C x H x W)
        """
        assert isinstance(
            images, (list, np.ndarray)
        ), "give images must be eather list of numpy arrays or numpy array"

        if isinstance(images, np.ndarray):
            images = [images]

        batch: List[torch.Tensor] = []

        for img in images:
            assert (
                len(img.shape) == 3
            ), "image shape must be channel, height\
                , with length of 3 but found {}".format(
                len(img.shape)
            )
            assert (
                img.shape[2] == 3
            ), "channel size of the image must be 3 but found {}".format(img.shape[2])

            batch.append(
                # h,w,c => c,h,w
                # pylint: disable=not-callable
                torch.tensor(img, dtype=self.dtype, device=self.device).permute(2, 0, 1)
            )

        return batch

    def to_json(self, preds: List[torch.Tensor]) -> List[Dict]:
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
                boxes = []
                scores = []

            results.append({"boxes": boxes, "scores": scores})

        return results
