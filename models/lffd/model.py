import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List,Dict
from .conv import conv_layer
from .resblock import ResBlock
from .detection import DetectionHead

from utils.utils import random_sample_selection

import math
import numpy as np
from cv2 import cv2

class LFFD(nn.Module):
    # these configs are for 8 headed lffd detector
    __FILTERS__ = [64,64,64,64,128,128,128,128]
    __RF_SIZES__ = [55,71,111,143,223,383,511,639]
    __RF_STRIDES__ = [4,4,8,8,16,32,32,32]
    __SCALES__ = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)] # calculated for 640 image input

    def __init__(self, in_channels:int=3, filters:List[int]=None,
            rf_sizes:List[int]=None, rf_strides:List[int]=None, scales:List[int]=None,
            num_classes:int=1, hyp:Dict={}, debug:bool=False):
        super(LFFD,self).__init__()

        if filters is None: filters = LFFD.__FILTERS__
        if rf_sizes is None: rf_sizes = LFFD.__RF_SIZES__
        if rf_strides is None: rf_strides = LFFD.__RF_STRIDES__
        if scales is None: scales = LFFD.__SCALES__
        self.num_classes = num_classes
        self.hyp = hyp
        self.__debug = debug

        # TODO check if list lenghts are matched

        # *tiny part
        self.downsample_conv1 = conv_layer(in_channels,64,stride=2,padding=1)
        self.downsample_conv2 = conv_layer(64,64,stride=2,padding=1)
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)
        self.res_block4 = ResBlock(64)

        # *small part
        self.downsample_conv3 = conv_layer(64,64,stride=2,padding=1)
        self.res_block4 = ResBlock(64)
        self.res_block5 = ResBlock(64)

        # *medium part
        self.downsample_conv4 = conv_layer(64,128,stride=2,padding=1)
        self.res_block6 = ResBlock(128)

        # *large part
        self.downsample_conv5 = conv_layer(128,128,stride=2,padding=1)
        self.res_block7 = ResBlock(128)
        self.res_block8 = ResBlock(128)
        self.res_block9 = ResBlock(128)

        self.heads = nn.ModuleList([
            DetectionHead(idx+1,num_of_filters,rf_size,rf_stride,lower_scale,upper_scale,
                num_classes=num_classes)
            for idx,(num_of_filters, rf_size, rf_stride, (lower_scale,upper_scale)) in enumerate(zip(filters,rf_sizes,rf_strides,scales))
        ])

    def forward(self, x:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits:List = []
        # *tiny part
        c1 = self.downsample_conv1(x)
        c2 = self.downsample_conv2(c1)
        c4 = self.res_block1(c2)
        c6 = self.res_block2(c4)
        c8 = self.res_block3(c6)
        logits.append(c8)

        c10 = self.res_block4(c8)
        logits.append(c10)

        # *small part
        c11 = self.downsample_conv3(c10)
        c13 = self.res_block4(c11)
        logits.append(c13)

        c15 = self.res_block5(c13)
        logits.append(c15)

        # *medium part
        c16 = self.downsample_conv4(c15)
        c18 = self.res_block6(c16)
        logits.append(c18)

        # *large part
        c19 = self.downsample_conv5(c18)
        c21 = self.res_block7(c19)
        logits.append(c21)

        c23 = self.res_block8(c21)
        logits.append(c23)

        c25 = self.res_block9(c23)
        logits.append(c25)

        cls_logits:List = []
        reg_logits:List = []

        for i in range(len(logits)):
            cls_l,reg_l = self.heads[i](logits[i])
            cls_logits.append(cls_l)
            reg_logits.append(reg_l)

        return cls_logits,reg_logits

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int) -> torch.Tensor:

        imgs,gt_boxes = batch

        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)
        ratio = 10

        cls_logits,reg_logits = self(imgs)

        target_cls:List = []
        target_regs:List = []

        target_objectness_mask:List = []
        target_reg_mask:List = []
        ignore:List = []

        debug_fmap = [] # ? Debug
        current = 0

        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]

            # ? Debug
            start = current
            current += (fh*fw)
            debug_fmap.append((fh,fw,start))

            t_cls,t_regs,ig = self.heads[i].build_targets((fh,fw), gt_boxes, device=device, dtype=dtype)
            # t_cls          : bs,fh',fw'      | type: model.dtype          | device: model.device
            # t_regs         : bs,fh',fw',4    | type: model.dtype          | device: model.device
            # ig             : bs,fh',fw'      | type: torch.bool           | device: model.device

            cls_logits[i] = cls_logits[i].permute(0,2,3,1).view(batch_size, -1)
            reg_logits[i] = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)

            target_cls.append(t_cls.view(batch_size, -1))
            target_regs.append(t_regs.view(batch_size,-1,4))

            ignore.append(ig.view(batch_size,-1))

        cls_logits = torch.cat(cls_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)
        target_cls = torch.cat(target_cls, dim=1)
        target_regs = torch.cat(target_regs, dim=1)
        ignore = torch.cat(ignore, dim=1)

        pos_mask = (target_cls == 1) & (~ignore)
        neg_mask = (target_cls == 0) & (~ignore)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        negatives = min(negatives,ratio*positives)

        # TODO change here to OHMN
        neg_mask = neg_mask.view(-1)
        ss, = torch.where(neg_mask)
        selections = random_sample_selection(ss.cpu().numpy().tolist(), negatives)
        neg_mask[:] = False
        neg_mask[selections] = True
        neg_mask = neg_mask.view(batch_size,-1)
        negatives = neg_mask.sum()
        ###########################

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[pos_mask | neg_mask], target_cls[pos_mask | neg_mask])

        reg_loss = F.mse_loss(reg_logits[pos_mask, :], target_regs[pos_mask, :])

        ## ? Debug
        if self.__debug:
            debug(imgs, gt_boxes, debug_fmap, self.heads, pos_mask, neg_mask, ignore)

        assert not torch.isnan(reg_loss) and not torch.isnan(cls_loss)

        return cls_loss + reg_loss

    def validation_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int) -> Dict:

        imgs,gt_boxes = batch

        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)
        ratio = 10

        cls_logits,reg_logits = self(imgs)

        target_cls:List = []
        target_regs:List = []

        target_objectness_mask:List = []
        target_reg_mask:List = []
        ignore:List = []
        preds:List = []

        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]

            t_cls,t_regs,ig = self.heads[i].build_targets((fh,fw), gt_boxes, device=device, dtype=dtype)
            # t_cls          : bs,fh',fw'      | type: model.dtype          | device: model.device
            # t_regs         : bs,fh',fw',4    | type: model.dtype          | device: model.device
            # ig             : bs,fh',fw'      | type: torch.bool           | device: model.device

            cls_logits[i] = cls_logits[i].permute(0,2,3,1).view(batch_size, -1)
            reg_logits[i] = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)

            with torch.no_grad():
                scores = torch.sigmoid(cls_logits[i].view(batch_size,fh,fw,1))
                pred_boxes = self.heads[i].apply_bbox_regression(
                    reg_logits[i].view(batch_size,fh,fw,4))

                pred_boxes = torch.cat([pred_boxes,scores], dim=-1).view(batch_size,-1,5)
                # pred_boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
                preds.append(pred_boxes)
                

            target_cls.append(t_cls.view(batch_size, -1))
            target_regs.append(t_regs.view(batch_size,-1,4))

            ignore.append(ig.view(batch_size,-1))

        preds = torch.cat(preds, dim=1)
        cls_logits = torch.cat(cls_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)
        target_cls = torch.cat(target_cls, dim=1)
        target_regs = torch.cat(target_regs, dim=1)
        ignore = torch.cat(ignore, dim=1)

        pos_mask = (target_cls == 1) & (~ignore)
        neg_mask = (target_cls == 0) & (~ignore)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        negatives = min(negatives,ratio*positives)

        # TODO change here to OHMN
        neg_mask = neg_mask.view(-1)
        ss, = torch.where(neg_mask)
        selections = random_sample_selection(ss.cpu().numpy().tolist(), negatives)
        neg_mask[:] = False
        neg_mask[selections] = True
        neg_mask = neg_mask.view(batch_size,-1)
        negatives = neg_mask.sum()
        ###########################

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[pos_mask | neg_mask], target_cls[pos_mask | neg_mask])

        reg_loss = F.mse_loss(reg_logits[pos_mask, :], target_regs[pos_mask, :])

        assert not torch.isnan(reg_loss) and not torch.isnan(cls_loss)
        loss = cls_loss + reg_loss

        pred_boxes:List = []
        for i in range(batch_size):
            pick = preds[i][:, 4] > 0.01
            pred_boxes.append(preds[i][pick])

        return {'loss':loss, 'preds': pred_boxes, 'gts': gt_boxes}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hyp.get('learning_rate',1e-2),
            momentum=self.hyp.get('momentum',0.9),
            weight_decay=self.hyp.get('weight_decay',0))

def debug(imgs, gt_boxes, debug_fmap, heads, pos_mask, neg_mask, ignore):
    for i,img in enumerate(imgs):
        nimg = (img*127.5 + 127.5).permute(1,2,0).cpu().numpy().astype(np.uint8)
        #nimg = cv2.UMat(nimg).get()
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        print(f"positives: {pos_mask[i].sum()} negatives:{neg_mask[i].sum()} ignore:{ignore[i].sum()}")
        for x1,y1,x2,y2 in gt_boxes[i].cpu().numpy().astype(np.int32):
            nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2), (255,0,0))
        for head,(fh,fw,start) in zip(heads,debug_fmap):
            priors = head.gen_rf_anchors(fh,fw,clip=True).view(-1,4)

            p_mask = pos_mask[i][start: start+fh*fw]
            n_mask = neg_mask[i][start: start+fh*fw]
            i_mask = ignore[i][start: start+fh*fw]

            pos_boxes = priors[p_mask].cpu().numpy().astype(np.int32).tolist()
            neg_boxes = priors[n_mask].cpu().numpy().astype(np.int32).tolist()
            ignore_boxes = priors[i_mask].cpu().numpy().astype(np.int32).tolist()

            for x1,y1,x2,y2 in pos_boxes:
                cx = int((x1+x2) / 2)
                cy = int((y1+y2) / 2)
                nimg = cv2.circle(nimg, (cx,cy), 2, (0,255,0))
            for x1,y1,x2,y2 in neg_boxes:
                cx = int((x1+x2) / 2)
                cy = int((y1+y2) / 2)
                nimg = cv2.circle(nimg, (cx,cy), 2, (0,0,255))
            for x1,y1,x2,y2 in ignore_boxes:
                cx = int((x1+x2) / 2)
                cy = int((y1+y2) / 2)
                nimg = cv2.circle(nimg, (cx,cy), 2, (0,255,255))

        cv2.imshow("",nimg)
        if cv2.waitKey(0) == 27:
            exit(0)