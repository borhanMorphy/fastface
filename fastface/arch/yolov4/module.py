from typing import Tuple, List, Dict
import torch
import torch.nn as nn

from .blocks import (
    CSPDarknet53Tiny,
    PANetTiny,
    YoloHead
)
from .anchor import Anchor

from ...loss import get_loss_by_name
from ...utils.box import batched_nms

class YOLOv4(nn.Module):
    __CONFIGS__ = {
        "tiny":{
            "input_shape": (-1, 3, 416, 416),
            "strides": [32, 16],
            "anchors": [
                [
                    [0.19471154, 0.19711538],
                    [0.32451923, 0.40625   ],
                    [0.82692308, 0.76682692]
                ],
                [
                    [0.02403846, 0.03365385],
                    [0.05528846, 0.06490385],
                    [0.08894231, 0.13942308]
                ]
            ],
            'head_infeatures': [512, 256],
            'neck_features': 512
        }
    }

    @staticmethod
    def get_anchor_generators(config: str) -> List[nn.Module]:
        assert config in YOLOv4.__CONFIGS__, "given config: {} not valid".format(config)
        config = YOLOv4.__CONFIGS__[config].copy()
        anchor_sizes = config['anchors']
        strides = config['strides']
        img_size = config['input_shape'][-1]

        anchors = []
        for _anchor_sizes, stride in zip(anchor_sizes, strides):
            anchors.append(Anchor(_anchor_sizes, img_size, stride))
        return anchors

    def __init__(self, config: Dict = {}, **kwargs):
        super().__init__()

        assert "input_shape" in config, "`input_shape` must be defined in the config"
        assert "strides" in config, "`strides` must be defined in the config"
        assert "anchors" in config, "`anchors` must be defined in the config"
        assert "head_infeatures" in config, "`head_infeatures` must be defined in the config"
        assert "neck_features" in config, "`neck_features` must be defined in the config"

        # TODO consider another model that is not tiny

        anchors = config['anchors']
        strides = config['strides']
        input_shape = config['input_shape']
        img_size = max(input_shape[-1], input_shape[-2])
        head_infeatures = config['head_infeatures']
        neck_features = config['neck_features']

        self.input_shape = input_shape
        self.backbone = CSPDarknet53Tiny()
        self.neck = PANetTiny(neck_features)
        self.heads = nn.ModuleList([
            YoloHead(in_features, img_size, stride=stride, anchors=_anchors)
            for stride, _anchors, in_features in zip(strides, anchors, head_infeatures)
        ])

        # TODO fix here
        self.cls_loss_fn = get_loss_by_name("BFL")
        self.reg_loss_fn = get_loss_by_name("MSE")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """preprocessed image batch

        Args:
            x (torch.Tensor): B x C x H x W

        Returns:
            List[torch.Tensor]: logits: List[B x 5 x H x W]
        """
        out, residual = self.backbone(x)
        outs = self.neck(out, residual)
        return [self.heads[i](out) for i, out in enumerate(outs)]

    def predict(self, x: torch.Tensor):
        """preprocessed image batch

        Args:
            x (torch.Tensor): B x C x H x W

        Returns:
            torch.Tensor: B x N x 5 as xmin, ymin, xmax, ymax, score
        """

        batch_size = x.size(0)

        head_logits = self.forward(x)

        batch_preds: List[torch.Tensor] = []

        for i, logits in enumerate(head_logits):
            # logits: b x Na x gridy x gridx x (4+1)

            logits[:, :, :, :, :4] = self.heads[i].det_layer.anchor.logits_to_boxes(logits[:, :, :, :, :4])

            logits[:, :, :, :, 4:] = torch.sigmoid(logits[:, :, :, :, 4:])

            # TODO fix -1
            batch_preds.append(logits.reshape(batch_size, -1, 5))

        # batch_preds: b x -1 x (4+1)
        return torch.cat(batch_preds, dim=1)

    def training_step(self, batch: Tuple[torch.Tensor, Dict],
            batch_idx: int, **hparams) -> torch.Tensor:

        # get these from hyper parameters
        reg_loss_weight = 1.0#0.07
        cls_loss_weight = 1.0

        imgs, targets = batch
        device = imgs.device
        dtype = imgs.dtype

        head_targets = targets['heads']

        """
        # head_targets;
            {
                'target_objectness':   bs x nA x nGy x nGx
                'ignore_objectness':   bs x nA x nGy x nGx
                'target_regs':         bs x nA x nGy x nGx x 4
            }

        """
        batch_size = imgs.size(0)
        logits = self.forward(imgs)

        losses = []

        for head_idx, head_logits in enumerate(logits):
            # head_logits: bs x nA x gy x gx x (4 + 1)

            target_objectness = head_targets[head_idx]['target_objectness'].to(device, dtype)
            ignore_objectness = head_targets[head_idx]['ignore_objectness'].to(device)
            target_regs = head_targets[head_idx]['target_regs'].to(device, dtype)

            pred_objectness = head_logits[:, :, :, :, 4]
            # pred_objectness: bs x nA x gy x gx

            pred_regressions = head_logits[:, :, :, :, :4]
            # pred_regressions: bs x nA x gy x gx x 4

            # loss objectness
            cls_loss = self.cls_loss_fn(
                pred_objectness[~ignore_objectness],
                target_objectness[~ignore_objectness]) * cls_loss_weight

            positive_mask = (target_objectness==1) & (~ignore_objectness)

            # loss regression
            reg_loss = self.reg_loss_fn(
                pred_regressions[positive_mask, :],
                target_regs[positive_mask, :]) * reg_loss_weight

            losses.append(cls_loss+reg_loss)

        return sum(losses)

    def validation_step(self, batch: Tuple[torch.Tensor, Dict],
            batch_idx: int, **hparams) -> Dict:

        # get these from hyper parameters
        reg_loss_weight = 1.0#0.07
        cls_loss_weight = 1.0

        det_threshold = 0.2
        iou_threshold = 0.4
        keep_top_n = 200

        imgs, targets = batch
        device = imgs.device
        dtype = imgs.dtype

        head_targets = targets['heads']
        gt_boxes = targets['gt_boxes']

        losses = []

        batch_size = imgs.size(0)
        logits = self.forward(imgs)

        batch_preds: List[torch.Tensor] = []

        for head_idx, head_logits in enumerate(logits):
            # logits: b x Na x gridy x gridx x (4+1)
            # head_logits: bs x nA x gy x gx x (4 + 1)

            target_objectness = head_targets[head_idx]['target_objectness'].to(device,dtype)
            ignore_objectness = head_targets[head_idx]['ignore_objectness'].to(device)
            target_regs = head_targets[head_idx]['target_regs'].to(device,dtype)

            pred_boxes = self.heads[head_idx].det_layer.anchor.logits_to_boxes(
                    head_logits[:, :, :, :, :4].cpu())

            pred_regressions = head_logits[:, :, :, :, :4]

            pred_objectness = head_logits[:, :, :, :, 4]

            batch_preds.append(
                torch.cat([
                    pred_boxes.reshape(batch_size, -1, 4),
                    torch.sigmoid(pred_objectness.reshape(batch_size, -1)).unsqueeze(-1).cpu()
                ], dim=-1)
            )

            # loss objectness
            cls_loss = self.cls_loss_fn(
                pred_objectness[~ignore_objectness],
                target_objectness[~ignore_objectness]) * cls_loss_weight

            positive_mask = (target_objectness==1) & (~ignore_objectness)

            # loss regression
            reg_loss = self.reg_loss_fn(
                pred_regressions[positive_mask, :],
                target_regs[positive_mask, :]) * reg_loss_weight

            losses.append(cls_loss+reg_loss)

        batch_preds = torch.cat(batch_preds, dim=1)
        # batch_preds: b x N x 5

        pick_b, pick_n = torch.where(batch_preds[:, :, 4] >= det_threshold)

        batch_preds, batch_ids = batched_nms(
            batch_preds[pick_b, pick_n, :4],
            batch_preds[pick_b, pick_n, 4],
            pick_b,
            iou_threshold
        )

        _batch_preds = []
        for i in range(batch_size):
            p = batch_preds[batch_ids == i]
            if p.size(0) == 0:
                _batch_preds.append(p)
                continue

            p = p[p[:, 4].argsort(descending=True), :]
            _batch_preds.append(p[:keep_top_n, :])

        return {
            'loss': sum(losses),
            'preds': _batch_preds,
            'gts': [gt[:, :4].cpu() for gt in gt_boxes]
        }

    def test_step(self, batch: Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs, targets = batch

        preds = self.predict(imgs)

        return {
            'preds': preds
        }

    def configure_optimizers(self, **hparams):
        # TODO
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=hparams.get("milestones", [600000, 1000000, 1200000, 1400000]),
            gamma=hparams.get("gamma", 0.1))

        return optimizer#[optimizer], [lr_scheduler]