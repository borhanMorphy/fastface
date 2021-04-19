from typing import Tuple, List, Dict

import torch
import torch.nn as nn

from ...loss import get_loss_by_name

from .blocks import (
    LFFDBackboneV1,
    LFFDBackboneV2,
    DetectionHead
)
from .anchor import Anchor

"""
self.cls_loss_fn = get_loss_by_name("BCE", negative_selection_rule="mix")
self.reg_loss_fn = get_loss_by_name("MSE")
"""

class LFFD(nn.Module):

    __CONFIGS__ = {
        'original':{
            'input_shape': (-1, 3, 640, 640),
            'backbone_name': 'lffd-v1',
            'head_infeatures': [64, 64, 64, 64, 128, 128, 128, 128],
            'head_outfeatures': [128, 128, 128, 128, 128, 128, 128, 128],
            'rf_sizes': [15, 20, 40, 70, 110, 250, 400, 560],
            'rf_start_offsets': [3, 3, 7, 7, 15, 31, 31, 31],
            'rf_strides': [4, 4, 8, 8, 16, 32, 32, 32],
            'scales': [
                (10, 15), (15, 20), (20, 40), (40, 70),
                (70, 110), (110, 250), (250, 400), (400, 560)
            ]
        },

        'slim':{
            'input_shape': (-1, 3, 480, 480),
            'backbone_name': 'lffd-v2',
            'head_infeatures': [64, 64, 64, 128, 128],
            'head_outfeatures': [128, 128, 128, 128, 128],
            'rf_sizes': [20, 40, 80, 160, 320],
            'rf_start_offsets': [3, 7, 15, 31, 63],
            'rf_strides': [4, 8, 16, 32, 64],
            'scales': [
                (10, 20), (20, 40), (40, 80), (80, 160), (160, 320)
            ]
        }
    }

    @staticmethod
    def get_anchor_generators(config: str) -> List[nn.Module]:
        assert config in LFFD.__CONFIGS__, "given config: {} not valid".format(config)
        config = LFFD.__CONFIGS__[config].copy()
        rf_strides = config['rf_strides']
        rf_start_offsets = config['rf_start_offsets']
        rf_sizes = config['rf_sizes']
        anchors = []
        for rf_stride, rf_start_offset, rf_size in zip(rf_strides, rf_start_offsets, rf_sizes):
            anchors.append(Anchor(rf_stride, rf_start_offset, rf_size))
        return anchors

    def __init__(self, in_channels:int=3, config:Dict={},
            debug:bool=False, **kwargs):
        super().__init__()

        assert "input_shape" in config, "`input_shape` must be defined in the config"
        assert "backbone_name" in config, "`backbone_name` must be defined in the config"
        assert "head_infeatures" in config, "`head_infeatures` must be defined in the config"
        assert "head_outfeatures" in config, "`head_outfeatures` must be defined in the config"
        assert "rf_sizes" in config, "`rf_sizes` must be defined in the config"
        assert "rf_start_offsets" in config, "`rf_start_offsets` must be defined in the config"
        assert "rf_strides" in config, "`rf_strides` must be defined in the config"
        assert "scales" in config, "`scales` must be defined in the config"

        backbone_name = config.get('backbone_name')
        head_infeatures = config.get('head_infeatures')
        head_outfeatures = config.get('head_outfeatures')
        rf_sizes = config.get('rf_sizes')
        rf_start_offsets = config.get('rf_start_offsets')
        rf_strides = config.get('rf_strides')

        self.input_shape = config.get('input_shape')

        # TODO check if list lenghts are matched
        if backbone_name == "lffd-v1":
            self.backbone = LFFDBackboneV1(in_channels)
        elif backbone_name == "lffd-v2":
            self.backbone = LFFDBackboneV2(in_channels)
        else:
            raise ValueError(f"given backbone name: {backbone_name} is not valid")

        self.heads = nn.ModuleList([
            DetectionHead(idx+1,infeatures,outfeatures, rf_size, rf_start_offset, rf_stride,
                num_classes=1)
            for idx,(infeatures,outfeatures, rf_size, rf_start_offset, rf_stride) in enumerate(zip(
                head_infeatures,head_outfeatures,rf_sizes,rf_start_offsets,rf_strides))
        ])

        self.cls_loss_fn = None
        self.reg_loss_fn = None

    def forward(self, x:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """preprocessed image batch

        Args:
            x (torch.Tensor): B x C x H x W

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                cls_logits: List[B x 1 x H x W]
                reg_logits: List[B x 4 x H x W]
        """
        logits = self.backbone(x)

        cls_logits: List[torch.Tensor] = []
        reg_logits: List[torch.Tensor] = []

        for i,head in enumerate(self.heads):
            cls_l,reg_l = head(logits[i])
            cls_logits.append(cls_l)
            reg_logits.append(reg_l)

        return cls_logits, reg_logits

    def predict(self, x:torch.Tensor) -> torch.Tensor:
        """preprocessed image batch

        Args:
            x (torch.Tensor): B x C x H x W

        Returns:
            torch.Tensor: B x N x 5 as xmin, ymin, xmax, ymax, score
        """
        batch_size = x.size(0)
        cls_logits, reg_logits = self.forward(x)
        preds:List[torch.Tensor] = []
        for i,head in enumerate(self.heads):
            # *for each head
            fh, fw = cls_logits[i].shape[2:]

            cls_ = cls_logits[i].permute(0,2,3,1).view(batch_size, fh*fw, 1)
            reg_ = reg_logits[i].permute(0,2,3,1).view(batch_size, fh*fw, 4)

            scores = torch.sigmoid(cls_.view(batch_size,fh,fw,1))
            # original positive pred score dim is 0
            boxes = head.anchor.logits_to_boxes( # ! TODO this line is broken for onnx deployment !
                reg_.view(batch_size,fh,fw,4))

            boxes = torch.cat([boxes,scores], dim=3).view(batch_size, fh*fw, 5)
            # boxes: bs,(fh*fw),5 as xmin, ymin, xmax, ymax, score
            preds.append(boxes)
        return torch.cat(preds, dim=1)

    def training_step(self, batch:Tuple[torch.Tensor, List],
            batch_idx:int, **hparams) -> torch.Tensor:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype

        """
        ## targets
        {
            "heads": [
                {
                    "target_cls":       fh',fw'    | torch.float,
                    "ignore_cls_mask":  fh',fw'    | torch.bool,
                    "target_regs":      fh', fw',4 | torch.float,
                    "reg_mask":         fh',fw'    | torch.bool
                }
            ],
            "gt_boxes": torch.Tensor
        }
        """

        num_of_heads = len(self.heads)
        heads_cls_logits,heads_reg_logits = self(imgs)
        head_losses:List = []
        heads = targets['heads']

        for i in range(num_of_heads):
            # *for each head
            target_cls = heads[i]['target_cls'].to(device,dtype)
            ignore_cls_mask = heads[i]['ignore_cls_mask'].to(device)
            target_regs = heads[i]['target_regs'].to(device,dtype)
            reg_mask = heads[i]['reg_mask'].to(device)

            cls_logits = heads_cls_logits[i].permute(0,2,3,1)
            reg_logits = heads_reg_logits[i].permute(0,2,3,1)

            _cls_logits = cls_logits[~ignore_cls_mask].squeeze()
            _target_cls = target_cls[~ignore_cls_mask]

            _reg_logits = reg_logits[reg_mask]
            _target_regs = target_regs[reg_mask]

            # pylint: disable=not-callable
            cls_loss = self.cls_loss_fn(_cls_logits, _target_cls)
            # pylint: disable=not-callable
            reg_loss = self.reg_loss_fn(_reg_logits, _target_regs)

            head_losses.append( cls_loss + reg_loss )

        return sum(head_losses)

    def validation_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)

        """
        ## targets
        {
            "heads": [
                {
                    "target_cls":       fh',fw'    | torch.float,
                    "ignore_cls_mask":  fh',fw'    | torch.bool,
                    "target_regs":      fh', fw',4 | torch.float,
                    "reg_mask":         fh',fw'    | torch.bool
                }
            ],
            "gt_boxes": torch.Tensor
        }
        """

        num_of_heads = len(self.heads)
        heads_cls_logits,heads_reg_logits = self(imgs)
        head_losses:List = []
        heads = targets['heads']

        preds:List = []

        for i in range(num_of_heads):
            # *for each head
            target_cls = heads[i]['target_cls'].to(device,dtype)
            ignore_cls_mask = heads[i]['ignore_cls_mask'].to(device)
            target_regs = heads[i]['target_regs'].to(device,dtype)
            reg_mask = heads[i]['reg_mask'].to(device)

            cls_logits = heads_cls_logits[i].permute(0,2,3,1)
            reg_logits = heads_reg_logits[i].permute(0,2,3,1)

            pred_boxes = self.heads[i].anchor.logits_to_boxes(reg_logits)

            scores = torch.sigmoid(cls_logits)
            pred_boxes = torch.cat([pred_boxes,scores], dim=-1).view(batch_size,-1,5)
            # pred_boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
            preds.append(pred_boxes)

            _cls_logits = cls_logits[~ignore_cls_mask].squeeze()
            _target_cls = target_cls[~ignore_cls_mask]

            _reg_logits = reg_logits[reg_mask]
            _target_regs = target_regs[reg_mask]
            # pylint: disable=not-callable
            cls_loss = self.cls_loss_fn(_cls_logits, _target_cls)
            # pylint: disable=not-callable
            reg_loss = self.reg_loss_fn(_reg_logits, _target_regs)

            head_losses.append(cls_loss+reg_loss)

        return {
            'loss': sum(head_losses),
            'preds': torch.cat(preds, dim=1)
        }

    def test_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,_ = batch
        
        preds = self.predict(imgs)

        return {
            'preds': preds,
        }

    def configure_optimizers(self, **hparams):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=hparams.get('learning_rate', 1e-1),
            momentum=hparams.get('momentum', 0.9),
            weight_decay=hparams.get('weight_decay', 1e-5))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=hparams.get("milestones", [600000, 1000000, 1200000, 1400000]),
            gamma=hparams.get("gamma", 0.1))

        return [optimizer], [lr_scheduler]