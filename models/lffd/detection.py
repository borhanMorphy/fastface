import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv_layer
from utils.matcher import LFFDMatcher
from utils.utils import random_sample_selection

class DetectionHead(nn.Module):
    def __init__(self, head_idx:int, features:int, rf_size:int, rf_stride:int,
            lower_scale:int, upper_scale:int, num_classes:int=1, num_target_reg:int=4):

        super(DetectionHead,self).__init__()
        self.head_idx = head_idx
        self.rf_size = rf_size
        self.rf_stride = rf_stride

        self.cls_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_classes,kernel_size=1,padding=0))

        self.reg_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_target_reg,kernel_size=1,padding=0))

        self.matcher = LFFDMatcher(lower_scale,upper_scale)

    def gen_rf_anchors(self, fmaph:int, fmapw:int, device:str='cpu') -> torch.Tensor:
        """takes feature map h and w and reconstructs rf anchors as tensor

        Args:
            fmaph (int): featuremap hight
            fmapw (int): featuremap width

        Returns:
            torch.Tensor: rf anchors as fmaph x fmapw x 4 (xmin, ymin, xmax, ymax)
        """
        # y: fmaph x fmapw
        # x: fmaph x fmapw
        y,x = torch.meshgrid(
            torch.arange(fmaph, dtype=torch.float32), torch.arange(fmapw, dtype=torch.float32))

        # rfs: fmaph x fmapw x 2
        rfs = torch.stack([x,y], dim=-1) + 0.5

        rfs *= self.rf_stride

        # rfs: fh x fw x 2 as x,y
        rfs = rfs.repeat(1,1,2) # fh x fw x 2 => fh x fw x 4
        rfs[:,:,:2] = rfs[:,:,:2] - self.rf_size/2
        rfs[:,:,2:] = rfs[:,:,2:] + self.rf_size/2

        return rfs.to(device)

    def compute_loss(self, logits:Tuple[torch.Tensor,torch.Tensor],
            batch_gt_boxes:List[torch.Tensor]) -> torch.Tensor:
        """Builds targets with given logits and ground truth boxes.

        Args:
            logits (Tuple(torch.Tensor,torch.Tensor)):
                cls_logits: b x fh x fw x 1
                reg_logits: b x fh x fw x 4
            batch_gt_boxes (List[torch.Tensor]): list of tensors :,4 as x1,y1,x2,y2

        Returns:

        """

        cls_logits,reg_logits = logits

        batch_size,fh,fw,_ = cls_logits.shape

        device = cls_logits.device

        batched_cls_mask = torch.zeros((batch_size,fh,fw), dtype=torch.int32, device=device) - 2 # default is `ignore`

        batched_cls_targets = torch.zeros((batch_size,fh,fw,cls_logits.size(-1)), dtype=cls_logits.dtype, device=device)

        for i,(gt_boxes,cls_mask,cls_targets) in enumerate(zip(batch_gt_boxes,batched_cls_mask,batched_cls_targets)):

            # move to same device if not
            if gt_boxes.device == device: gt_boxes = gt_boxes.to(device)

            rf_anchors = self.gen_rf_anchors(fh,fw,device=device)

            cls_logit_mask,reg_logit_mask = self.matcher(rf_anchors, gt_boxes)

            batched_cls_mask[i] = cls_logit_mask
            batched_cls_targets[i, cls_logit_mask>=0] = 1
            batched_cls_targets[i, cls_logit_mask<0] = 0

        ratio = 10
        pos_sample_count = batched_cls_targets[batched_cls_targets==1].shape[0]
        neg_sample_count = ratio*pos_sample_count

        batched_cls_targets = batched_cls_targets.reshape(-1)
        cls_logits = cls_logits.reshape(-1)

        batched_cls_mask = batched_cls_mask.reshape(-1)

        neg_sample_count = min(neg_sample_count, batched_cls_mask[batched_cls_mask==-1].shape[0])
        if neg_sample_count == 0:
            return torch.tensor(0, dtype=cls_logits.dtype, requires_grad=True, device=cls_logits.device), None

        with torch.no_grad():
            loss = F.binary_cross_entropy_with_logits(
                cls_logits,
                batched_cls_targets,
                reduction='none')

        selectables, = torch.where(batched_cls_mask==-1)
        sorted_selections = loss.argsort(descending=True)
        neg_selections = []
        for candidate in sorted_selections:
            if candidate in selectables:
                neg_selections.append(candidate)
            if len(neg_selections) == neg_sample_count:
                break

        neg_selections = torch.tensor(neg_selections, dtype=torch.long, device=cls_logits.device)


        pos_selections, = torch.where(batched_cls_targets==1)

        selections = torch.cat([pos_selections,neg_selections], dim=0)
        loss = F.binary_cross_entropy_with_logits(cls_logits[selections],batched_cls_targets[selections])

        debug_data = (pos_selections, neg_selections, fh, fw, self.head_idx)

        return loss,debug_data

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        cls_logits = self.cls_head(input).permute(0,2,3,1).contiguous()
        # (b,c,h,w) => (b,h,w,c)
        reg_logits = self.reg_head(input).permute(0,2,3,1).contiguous()
        # (b,c,h,w) => (b,h,w,c)
        return cls_logits,reg_logits