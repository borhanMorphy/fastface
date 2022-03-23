import math
import os
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from cv2 import cv2
from torchmetrics.metric import Metric

from . import api, utils
from .config import ArchConfig


class FaceDetector(pl.LightningModule):
    """Generic pl.LightningModule definition for face detection"""

    def __init__(self, arch: nn.Module = None):
        super().__init__()
        self.arch = arch
        self.__metrics = dict()
        self.every_n_epoch = math.inf
        self.every_n_batch = math.inf

        self.init_preprocess(
            mean=arch.config.mean,
            std=arch.config.std,
            normalized_input=arch.config.normalized_input,
        )

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

    def metrics(self, name: str, metric: Metric):
        """Adds given metric with name key

        Args:
                name (str): name of the metric
                metric (Metric): Metric object
        """
        # TODO add warnings if override happens
        self.__metrics[name] = metric

    def get_metric(self, name: str) -> Metric:
        """Return metrics defined in the `FaceDetector` instance

        Returns:
                Metric: defined model metrics with names
        """
        # TODO
        return self.__metrics[name]

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
        >>> from cv2 import cv2
        >>> model = ff.FaceDetector.from_pretrained('lffd_original').eval()
        >>> img = cv2.imread('resources/friends.jpg')[..., [2, 1, 0]]
        >>> model.predict(img, target_size=640)
        [{'boxes': [[1057, 180, 1187, 352], [571, 212, 697, 393], [140, 218, 270, 382], [864, 271, 979, 406], [327, 252, 442, 392]], 'scores': [0.9992133378982544, 0.9971852898597717, 0.9246336817741394, 0.8549349308013916, 0.8072562217712402]}]
        """

        batch = utils.process.to_tensor(data)
        # batch: list of tensors

        batch_size = len(batch)

        batch, scales, paddings = utils.process.prepare_batch(
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

        preds = utils.process.adjust_results(preds, scales, paddings)
        # preds: list of torch.Tensor(N, 5) as x1,y1,x2,y2,score

        return utils.process.to_json(preds)

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
            torch.Tensor: preds with shape (N, 5 + num_landmarks*2 + 1);
                    (1:4) xmin, ymin, xmax, ymax
                    (4:5) score
                    (5:5+num_landmarks*2)
                    (5+num_landmarks*2:) batch idx
        """

        # apply preprocess
        batch = ((batch / self.normalizer) - self.mean) / self.std

        # get logits
        with torch.no_grad():
            logits = self.arch.forward(batch)
        # logits, any

        preds = self.arch.compute_preds(logits)
        # preds: torch.Tensor(B, N, 5)

        return self._postprocess(
            preds,
            det_threshold,
            iou_threshold,
            keep_n,
        )

    def _postprocess(
        self,
        preds: torch.Tensor,
        det_threshold: float,
        iou_threshold: float,
        keep_n: int,
    ) -> torch.Tensor:
        """Applies postprocess to given predictions

        Args:
            preds (torch.Tensor): predictions as B x N x (5 + num_landmarks*2)
            det_threshold (float): detection score threshold
            iou_threshold (float): iou score threshold for NMS
            keep_n (int): keep number of predictions

        Returns:
            torch.Tensor: predictions as N x (5 + 1)
        """
        batch_size = preds.size(0)

        # filter out some predictions using score
        pick_b, pick_n = torch.where(preds[:, :, 4] >= det_threshold)

        # add batch_idx dim to preds
        preds = torch.cat(
            [preds[pick_b, pick_n, :5], pick_b.to(preds.dtype).unsqueeze(1)], dim=1
        )
        # preds: N x (5 + 1)

        batch_preds: List[torch.Tensor] = []
        for batch_idx in range(batch_size):
            (pick_n,) = torch.where(batch_idx == preds[:, -1])
            order = preds[pick_n, 4].sort(descending=True)[1]

            batch_preds.append(
                # preds: N, 6
                preds[pick_n, :][order][:keep_n, :]
            )

        batch_preds = torch.cat(batch_preds, dim=0)
        # batch_preds: N' x (5 + 1)

        # filter with nms
        pick = utils.box.batched_nms(
            batch_preds[:, :4],  # boxes as x1,y1,x2,y2
            batch_preds[:, 4],  # det score between [0, 1]
            batch_preds[:, 5],  # id of the batch that prediction belongs to
            iou_threshold=iou_threshold,
        )
        return batch_preds[pick, :]

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

            if self.is_debug_step(batch_idx):
                for img, preds, gt_boxes in zip(batch[0], pred_boxes, target_boxes):
                    img = (img.permute(1, 2, 0).cpu()).numpy().astype(np.uint8).copy()
                    preds = preds.cpu().long().numpy()
                    gt_boxes = gt_boxes.cpu().long().numpy()

                    for x1, y1, x2, y2 in preds[:, :4]:
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
                    for x1, y1, x2, y2 in gt_boxes[:, :4]:
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

                    cv2.imshow("", img[..., [2, 1, 0]])

                    if cv2.waitKey(0) == 27:
                        exit(0)

        target_logits = self.arch.build_targets(inputs, raw_targets)

        # compute loss
        # loss: dict of losses or loss
        loss = self.arch.compute_loss(logits, target_logits)

        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log("{}/{}".format(key, phase), value.item())
        else:
            self.log("loss/{}".format(phase), loss.item())

        return loss

    def _on_epoch_start(self, phase: str = None):
        for metric in self.__metrics.values():
            metric.reset()

    def _epoch_end(self, _, phase: str = None):
        if phase != "train":
            for name, metric in self.__metrics.items():
                res = metric.compute()
                if isinstance(res, dict):
                    for k, v in res.items():
                        self.log("{}/{}/{}".format(name, phase, k), v)
                else:
                    self.log("{}/{}".format(name, phase), res)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="train")

    def on_validation_epoch_start(self):
        self._on_epoch_start(phase="validation")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="validation")

    def validation_epoch_end(self, _):
        return self._epoch_end(_, phase="validation")

    def on_test_epoch_start(self):
        self._on_epoch_start(phase="test")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, phase="test")

    def test_epoch_end(self, _):
        return self._epoch_end(_, phase="test")

    def configure_optimizers(self):
        return self.arch.configure_optimizers()

    @classmethod
    def build(
        cls,
        arch: str,
        config: Union[str, ArchConfig],
    ) -> pl.LightningModule:
        """Classmethod for creating `fastface.FaceDetector` instance from scratch

        Args:
                arch (str): architecture name
                config (Union[str, Dict]): configuration name or configuration dictionary

        Returns:
                pl.LightningModule: fastface.FaceDetector instance with random weights initialization
        """

        # build nn.Module with given configuration
        arch = api.build_arch(arch, config)

        # build pl.LightninModule with given architecture
        return cls(arch=arch)

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
        config = ArchConfig.construct(**checkpoint["configuration"])

        # build nn.Module with given configuration
        self.arch = api.build_arch(config.arch, config)

        # initialize preprocess with given arguments
        self.init_preprocess(
            mean=self.arch.config.mean,
            std=self.arch.config.std,
            normalized_input=self.arch.config.normalized_input,
        )

    def on_save_checkpoint(self, checkpoint: Dict):
        checkpoint["configuration"] = dict(self.arch.config)
