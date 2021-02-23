from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops

class YOLOv4(nn.Module):
    __CONFIGS__ = {
        'tiny':{
            
        }
    }

    _transforms = Compose(
        Interpolate(max_dim=416),
        Padding(target_size=(416,416)),
        Normalize(mean=127.5, std=127.5),
        ToTensor()
    )

    def __init__(self, in_channels:int=3, config:Dict={},
            num_classes:int=1, debug:bool=False, **kwargs):
        super(LFFD,self).__init__()

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

        self.nms = kwargs.get('nms', box_ops.nms)
        self.num_classes = num_classes
        self.__debug = debug

        # TODO check if list lenghts are matched
        if backbone_name == "tiny":
            self.backbone = None # TODO
        else:
            raise ValueError(f"given backbone name: {backbone_name} is not valid")

        self.cls_loss_fn = # TODO get_loss_by_name("BCE", negative_selection_rule="mix")
        self.reg_loss_fn = # TODO get_loss_by_name("MSE")

    def forward(self, x:torch.Tensor):
        pass # TODO

    @torch.no_grad()
    def predict(self, x:torch.Tensor, det_threshold:float=.95,
            iou_threshold:float=.4):
        pass # TODO

    def training_step(self, batch:Tuple[torch.Tensor, List],
            batch_idx:int, **hparams) -> torch.Tensor:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype

        """
        ## targets
        # TODO
        """
        # TODO

    def validation_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)

        """
        ## targets
        # TODO
        """

        return {
            'loss': None, # TODO
            'preds': None, # TODO
            'gts': None # TODO
        }

    def test_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)

        """
        ## targets
        # TODO
        """

        return {
            'preds': None, # TODO
            'gts': None # TODO
        }

    def configure_optimizers(self, **hparams):
        # TODO
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