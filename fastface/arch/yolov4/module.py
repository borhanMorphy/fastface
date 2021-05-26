from typing import List, Dict, Tuple
import torch
import torch.nn as nn

from .blocks import (
    CSPDarknet53Tiny,
    PANetTiny,
    YoloHead
)

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
        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_loss_fn = nn.MSELoss(reduction='none')

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

    def logits_to_preds(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Applies postprocess to given logits

        Args:
            logits (List[torch.Tensor]): list of logits as B x FH x FW x 5
                (0:4) reg logits
                (4:5) cls logits
        Returns:
            torch.Tensor: as preds with shape of B x N x 5 where x1,y1,x2,y2,score
        """
        # TODO implement here
        raise NotImplementedError

    def compute_loss(self, logits: List[torch.Tensor],
            raw_targets: List[Dict], hparams: Dict = {}) -> Dict[str, torch.Tensor]:
        """Computes loss using given logits and raw targets

        Args:
            logits (List[torch.Tensor]): list of torch.Tensor(B, fh, fw, 5) where;
                (0:4) reg logits
                (4:5) cls logits
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            hparams (Dict, optional): model hyperparameter dict. Defaults to {}.

        Returns:
            Dict[str, torch.Tensor]: loss values as key value pairs

        """
        # TODO implement here
        raise NotImplementedError

    def build_targets(self, fmap_shapes: List[Tuple[int, int]], raw_targets: List[Dict],
            dtype=torch.float32, device="cpu") -> torch.Tensor:
        """build model targets using given logits and raw targets

        Args:
            fmap_shapes (List[Tuple[int, int]]): feature map shapes for each head, [(fh,fw), ...]
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            dtype : dtype of the tensor. Defaults to torch.float32.
            device (str): device of the tensor. Defaults to 'cpu'.

        Returns:
            torch.Tensor: targets as B x N x 5 where (0:4) reg targets (4:5) cls targets
        """

        # TODO implement here
        raise NotImplementedError

    def configure_optimizers(self, **hparams):
        # TODO
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0)

        return optimizer
