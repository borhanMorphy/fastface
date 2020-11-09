import torch
import torch.nn as nn

from typing import Tuple,List
from .conv import conv_layer

class DetectionHead(nn.Module):
    def __init__(self, features:int, rf_size:int, rf_stride:int,
            lower_scale:int, upper_scale:int, num_classes:int=1, num_target_reg:int=4):

        super(DetectionHead,self).__init__()

        self.rf_size = rf_size
        self.rf_stride = rf_stride

        self.natural_rfs = []

        self.cls_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_classes,kernel_size=1,padding=0))

        self.reg_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_target_reg,kernel_size=1,padding=0))

    def gen_rf_centers(self, fmaph:int, fmapw:int, device:str='cpu') -> torch.Tensor:
        """takes feature map h and w and reconstructs rf centers as tensor

        Args:
            fmaph (int): featuremap hight
            fmapw (int): featuremap width

        Returns:
            torch.Tensor: rf centers as fmaph x fmapw x 2 (x, y order)
        """

        # y: fmaph x fmapw
        # x: fmaph x fmapw
        y,x = torch.meshgrid(
            torch.arange(fmaph, dtype=torch.float32), torch.arange(fmapw, dtype=torch.float32))

        # rfs: fmaph x fmapw x 2
        rfs = torch.stack([x,y], dim=-1) + 0.5

        rfs *= self.rf_stride

        return rfs.to(device)

    def gen_rf_anchors(self, fmaph:int, fmapw:int, device:str='cpu') -> torch.Tensor:
        """takes feature map h and w and reconstructs rf anchors as tensor

        Args:
            fmaph (int): featuremap hight
            fmapw (int): featuremap width

        Returns:
            torch.Tensor: rf anchors as fmaph x fmapw x 4 (xmin, ymin, xmax, ymax)
        """

        rfs = self.gen_rf_centers(fmaph, fmapw, device=device)
        # rfs: fh x fw x 2 as x,y
        rfs = rfs.repeat(1,1,2) # fh x fw x 2 => fh x fw x 4
        rfs[:,:,:2] = rfs[:,:,:2] - self.rf_size/2
        rfs[:,:,2:] = rfs[:,:,2:] + self.rf_size/2

        return rfs

    def build_targets(self, logits:Tuple[torch.Tensor,torch.Tensor],
            batch_gt_boxes:List[torch.Tensor], debug:bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds targets with given logits and ground truth boxes.

        Args:
            logits (Tuple(torch.Tensor,torch.Tensor)):
                cls_logits: b x 1 x fh x fw
                reg_logits: b x 4 x fh x fw
            batch_gt_boxes (List[torch.Tensor]): list of tensors :,4 as x1,y1,x2,y2

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                cls_logits  : N x 1
                reg_logits  : N x 4
                cls_targets : N x 1
                reg_targets : N x 4

        """

        batched_debug_information = []

        cls_logits,reg_logits = logits

        batch_size,_,fh,fw = cls_logits.shape

        cls_logits = cls_logits.permute(0,2,3,1).contiguous() # b x c x fh x fw => b x fh x fw x c
        reg_logits = reg_logits.permute(0,2,3,1).contiguous() # b x c x fh x fw => b x fh x fw x c

        device = cls_logits.device

        batched_cls_mask = torch.zeros((batch_size,fh,fw), dtype=torch.int32, device=device) - 2 # default is `ignore`
        batched_reg_mask = torch.zeros((batch_size,fh,fw), dtype=torch.int32, device=device) - 2 # default is `ignore`

        batched_cls_targets = torch.zeros((batch_size,fh,fw,cls_logits.size(-1)), dtype=cls_logits.dtype, device=device)
        batched_reg_targets = torch.zeros((batch_size,fh,fw,reg_logits.size(-1)), dtype=reg_logits.dtype, device=device)

        for i in range(batch_size):
            debug_information = []
            gt_boxes = batch_gt_boxes[i]

            # move to same device if not
            if gt_boxes.device == device: gt_boxes = gt_boxes.to(device)

            # TODO cache rfs
            rfs = self.gen_rf_centers(fh,fw,device=device)
            rf_anchors = self.gen_rf_anchors(fh,fw,device=device)

            # torch.Tensor: rf centers as fh x fw x 2 (x, y order)
            cls_selected_ids,reg_selected_ids = self.matcher(rfs,gt_boxes)
            """
            cls_selected_ids: fh x fw
            reg_selected_ids: fh x fw

            -1   : negative
            >= 0 : positive index
            -2   : ignored
            """
            batched_cls_mask[i][cls_selected_ids >= 0] = 1
            batched_cls_mask[i][cls_selected_ids == -1] = 0

            batched_reg_mask[i][reg_selected_ids >= 0] = 1
            batched_reg_mask[i][reg_selected_ids == -1] = 0

            batched_cls_targets[i][cls_selected_ids >= 0] = 1
            batched_cls_targets[i][cls_selected_ids == -1] = 0

            for j in range(len(gt_boxes)):
                # TODO convert to regression target
                if batched_reg_targets[i][reg_selected_ids == j].size(0) == 0: continue
                x1,y1,x2,y2 = gt_boxes[j]
                selected_rfs = rfs[reg_selected_ids == j]
                selected_rf_anchors = rf_anchors[reg_selected_ids == j]

                # M,2 as cx,cy
                t_regs = torch.zeros((selected_rfs.size(0),4), dtype=gt_boxes.dtype, device=gt_boxes.device)
                norm_const = self.rf_size / 2
                if debug:
                    debug_information.append((selected_rf_anchors,(x1,y1,x2,y2)))

                t_regs[:, 0] = (selected_rfs[:, 0] - x1) / norm_const
                t_regs[:, 1] = (selected_rfs[:, 1] - y1) / norm_const
                t_regs[:, 2] = (selected_rfs[:, 0] - x2) / norm_const
                t_regs[:, 3] = (selected_rfs[:, 1] - y2) / norm_const

                batched_reg_targets[i][reg_selected_ids == j] = t_regs

            if debug:
                batched_debug_information.append(debug_information)

        cls_cond = torch.bitwise_or(batched_cls_mask==1, batched_cls_mask==0)
        reg_cond = batched_reg_mask == 1

        cls_logits = cls_logits[cls_cond]
        reg_logits = reg_logits[reg_cond]

        cls_targets = batched_cls_targets[cls_cond]
        reg_targets = batched_reg_targets[reg_cond]

        if debug:
            return cls_logits, reg_logits, cls_targets, reg_targets, batched_debug_information

        return cls_logits, reg_logits, cls_targets, reg_targets

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        pred_cls = self.cls_head(input)
        reg_cls = self.reg_head(input)
        return pred_cls,reg_cls