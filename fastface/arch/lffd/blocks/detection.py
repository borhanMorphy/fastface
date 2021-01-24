import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv1x1
from fastface.utils.random import random_sample_selection
import math

class DetectionHead(nn.Module):
    def __init__(self, head_idx:int, infeatures:int, features:int, num_classes:int=1):
        super(DetectionHead,self).__init__()
        self.head_idx = head_idx
        self.num_classes = num_classes

        self.det_conv = nn.Sequential(
            conv1x1(infeatures, features), nn.ReLU())

        self.cls_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, self.num_classes))

        self.reg_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, 4))

        def conv_xavier_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(conv_xavier_init)

    def compute_loss(self, cls_items:Tuple[torch.Tensor,torch.Tensor,torch.Tensor],
            reg_items:Tuple[torch.Tensor,torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        # (cls_logits,target_cls,mask_cls),
        # (reg_logits,target_regs,mask_regs)
        cls_logits,target_cls,mask_cls = cls_items
        reg_logits,target_regs,mask_regs = reg_items

        device = reg_logits.device
        dtype = reg_logits.dtype

        bs,fh,fw = cls_logits.shape[:3]

        pos_mask = mask_cls & (target_cls == 1)
        neg_mask = mask_cls & (target_cls == 0)
        top_neg_mask = neg_mask.clone().view(-1)
        top_neg_mask[:] = False
        rand_neg_mask = neg_mask.clone().view(-1)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        hnm_ratio = 5
        random_ratio = 5
        neg_select_ratio = 0.1

        if positives > 0:
            top_negatives = min(positives*hnm_ratio, negatives//2)
            rand_negatives = min(positives*random_ratio, negatives//2)
        else:
            top_negatives = min(int((pos_mask.view(-1).size(0) // 2) * neg_select_ratio), negatives//2)
            rand_negatives = min(int((pos_mask.view(-1).size(0) // 2) * neg_select_ratio), negatives//2)

        _,top_neg_ids = cls_logits.view(-1).topk(top_negatives)
        top_neg_mask[top_neg_ids] = True
        top_neg_mask = top_neg_mask.view(bs,fh,fw)
        rand_neg_mask[top_neg_ids] = False
        rand_negatives = min(rand_negatives,rand_neg_mask.sum())

        rand_neg_ids, = torch.where(rand_neg_mask)
        pick = random_sample_selection(rand_neg_ids.cpu().numpy().tolist(), rand_negatives)
        rand_neg_mask[:] = False
        rand_neg_mask[pick] = True
        rand_neg_mask = rand_neg_mask.view(bs,fh,fw)
        neg_mask = rand_neg_mask | top_neg_mask

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[pos_mask | neg_mask].squeeze(), target_cls[pos_mask | neg_mask], reduction='none')

        if mask_regs.sum() > 0:
            reg_loss = F.mse_loss(
                reg_logits[mask_regs, :], target_regs[mask_regs, :], reduction='none')
        else:
            reg_loss = torch.tensor([[0,0,0,0]], dtype=dtype, device=device, requires_grad=True)

        return cls_loss,reg_loss

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.det_conv(x)

        cls_logits = self.cls_head(data)
        # (b,c,h,w)
        reg_logits = self.reg_head(data)
        # (b,c,h,w)
        return cls_logits,reg_logits