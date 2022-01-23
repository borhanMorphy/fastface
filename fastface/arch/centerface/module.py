from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ... import utils
from ...loss import BinaryFocalLoss
from .blocks import FPN, CenterFaceHead, MobileNetV2
from .utils import gaussian_radius


class CenterFace(nn.Module):

    __CONFIGS__ = {
        "original": {
            "input_shape": (-1, 3, 512, 512),
            "backbone": {"name": "mobilenetv2"},
            "fpn": {
                "in_features": [24, 32, 96, 320],
                "out_feature": 24,
                "upsample_method": "deconv",
            },
            "output_stride": 4,
            "cls_loss": {
                "gamma": 2,
                "alpha": 1,
                "beta": 4,
            },
        },
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__()

        # TODO config assertions

        if config["backbone"]["name"] == "mobilenetv2":
            # TODO
            backbone = MobileNetV2.from_pretrained()
        else:
            raise AssertionError("undefined backbone")

        self.backbone = backbone
        self.neck = FPN(
            config["fpn"]["in_features"],
            config["fpn"]["out_feature"],
            upsample_method=config["fpn"]["upsample_method"],
        )
        self.head = CenterFaceHead(config["fpn"]["out_feature"])

        self.output_stride = config["output_stride"]
        self.input_shape = config.get("input_shape")

        self.register_buffer(
            "grids",
            utils.box.generate_grids(
                1500 / self.output_stride, 1500 / self.output_stride
            ),
            persistent=False,
        )

        # define losses
        self.cls_loss = BinaryFocalLoss(
            gamma=config["cls_loss"]["gamma"], alpha=config["cls_loss"]["alpha"]
        )
        self.beta_val = config["cls_loss"]["beta"]

        self.offset_loss = nn.SmoothL1Loss(reduction="none")
        self.wh_loss = nn.L1Loss(reduction="none")

    def forward(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.neck(self.backbone(batch))
        return self.head(logits)

    def logits_to_preds(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Applies postprocess to given logits

        Args:
            logits (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): TODO
        Returns:
            torch.Tensor: as preds with shape of B x N x 5 where x1,y1,x2,y2,score
        """

        cls_logits, offset_regs, wh_regs = logits

        batch_size, fh, fw = cls_logits.shape
        heatmap = torch.sigmoid(cls_logits)

        centers = (
            self.grids[:fh, :fw, :].repeat(batch_size, 1, 1, 1) + 0.5 + offset_regs
        ) * self.output_stride
        wh = torch.exp(wh_regs) * self.output_stride

        x1y1 = centers - wh / 2
        x2y2 = x1y1 + wh

        return torch.cat([x1y1, x2y2, heatmap.unsqueeze(3)], dim=3).flatten(
            start_dim=1, end_dim=2
        )

    def compute_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        raw_targets: List[Dict],
        hparams: Dict = {},
    ) -> Dict[str, torch.Tensor]:

        lambda_offset = hparams.get("lambda_offset", 1.0)
        lambda_wh = hparams.get("lambda_wh", 0.1)

        cls_logits, offset_regs, wh_regs = logits

        _, _, fh, fw = offset_regs.shape

        target_cls, target_offset, target_wh = self.build_targets(
            max(fh, fw), raw_targets, dtype=offset_regs.dtype, device=offset_regs.device
        )

        mask = target_cls != 1
        num_positives = (~mask).sum()
        cls_loss = self.cls_loss(cls_logits, target_cls)
        cls_loss[num_positives:] = (
            torch.pow(1 - target_cls[mask], self.beta_val) * cls_loss[num_positives:]
        )

        cls_loss = cls_loss.sum() / num_positives

        offset_loss = (
            self.offset_loss(
                offset_regs[~mask, :],
                target_offset[~mask, :],
            ).sum()
            / num_positives
        )

        wh_loss = (
            self.wh_loss(
                wh_regs[~mask, :],
                target_wh[~mask, :],
            ).sum()
            / num_positives
        )

        loss = cls_loss + (offset_loss * lambda_offset) + (wh_loss * lambda_wh)

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "offset_loss": offset_loss.detach(),
            "wh_loss": wh_loss.detach(),
        }

    def build_targets(
        self,
        fmap_size: int,
        raw_targets: List[Dict],
        dtype=torch.float32,
        device="cpu",
    ):

        batch_size = len(raw_targets)

        target_cls = torch.zeros(
            *(batch_size, fmap_size, fmap_size), dtype=dtype, device=device
        )
        target_offset = torch.zeros(
            *(batch_size, fmap_size, fmap_size, 2), dtype=dtype, device=device
        )
        target_wh = torch.zeros(
            *(batch_size, fmap_size, fmap_size, 2), dtype=dtype, device=device
        )

        for i, targets in enumerate(raw_targets):
            target_boxes = targets["target_boxes"].to(device, dtype)
            centers = (target_boxes[:, [0, 1]] + target_boxes[:, [2, 3]]) / 2
            fmap_centers = (centers / self.output_stride).long()

            # set target cls
            for j, center in enumerate(fmap_centers):
                # center: x, y
                target_w = (
                    (targets["target_boxes"][j, 2] - targets["target_boxes"][j, 0])
                    .cpu()
                    .item()
                )
                target_h = (
                    (targets["target_boxes"][j, 3] - targets["target_boxes"][j, 1])
                    .cpu()
                    .item()
                )
                sigma = (
                    gaussian_radius((target_h, target_w), min_overlap=0.7) * 2 + 1
                ) / 6
                heatmap = utils.kernel.get_gaussian_kernel(
                    fmap_size,
                    sigma=sigma,
                    center_point=center.cpu().numpy(),
                    normalize=False,
                )
                heatmap = torch.from_numpy(heatmap).to(device, dtype)
                target_cls[i, ...] = torch.maximum(heatmap, target_cls[i, ...])
            # set target reg
            true_indexes_x, true_indexes_y = fmap_centers.t()
            w = target_boxes[:, 2] - target_boxes[:, 0]
            h = target_boxes[:, 3] - target_boxes[:, 1]

            target_wh[i, true_indexes_y, true_indexes_x, 0] = torch.log(
                w / self.output_stride + 1e-8
            )
            target_wh[i, true_indexes_y, true_indexes_x, 1] = torch.log(
                h / self.output_stride + 1e-8
            )

            # set target offset
            target_offset[i, true_indexes_y, true_indexes_x, :] = (
                centers / self.output_stride
            ) - fmap_centers

        return target_cls, target_offset, target_wh

    def configure_optimizers(self, hparams: Dict = {}):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=hparams.get("learning_rate", 5e-4)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=hparams.get("lr_milestones", [90, 120]),
            gamma=hparams.get("lr_gamma", 0.1),
        )
        return [optimizer], [scheduler]
