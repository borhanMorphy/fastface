import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from typing import Tuple,List,Dict
from .conv import conv3x3
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
        self.downsample_conv1 = nn.Sequential(
            conv3x3(in_channels,64,stride=2,padding=1), nn.ReLU6(inplace=True))
        self.downsample_conv2 = nn.Sequential(
            conv3x3(64,64,stride=2,padding=1), nn.ReLU6(inplace=True))
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)
        self.res_block4 = ResBlock(64)

        # *small part
        self.downsample_conv3 = nn.Sequential(
            conv3x3(64,64,stride=2,padding=1), nn.ReLU6(inplace=True))
        self.res_block4 = ResBlock(64)
        self.res_block5 = ResBlock(64)

        # *medium part
        self.downsample_conv4 = nn.Sequential(
            conv3x3(64,128,stride=2,padding=1), nn.ReLU6(inplace=True))
        self.res_block6 = ResBlock(128)

        # *large part
        self.downsample_conv5 = nn.Sequential(
            conv3x3(128,128,stride=2,padding=1), nn.ReLU6(inplace=True))
        self.res_block7 = ResBlock(128)
        self.res_block8 = ResBlock(128)
        self.res_block9 = ResBlock(128)

        self.heads = nn.ModuleList([
            DetectionHead(idx+1,infeatures,128,rf_size,rf_stride,lower_scale,upper_scale,
                num_classes=num_classes)
            for idx,(infeatures, rf_size, rf_stride, (lower_scale,upper_scale)) in enumerate(zip(filters,rf_sizes,rf_strides,scales))
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

    @torch.no_grad()
    def predict(self, x:torch.Tensor) -> List[torch.Tensor]:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        cls_logits,reg_logits = self.forward(x)
        preds:List = []
        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]

            cls_ = cls_logits[i].permute(0,2,3,1).view(batch_size, -1)
            reg_ = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)
            # TODO checkout here
            scores = torch.sigmoid(cls_.view(batch_size,fh,fw,1))
            boxes = self.heads[i].apply_bbox_regression(
                reg_.view(batch_size,fh,fw,4))

            boxes = torch.cat([boxes,scores], dim=-1).view(batch_size,-1,5)
            # boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
            preds.append(boxes)
        preds = torch.cat(preds, dim=1)

        pred_boxes:List = []
        for i in range(batch_size):
            selected_boxes = preds[i, preds[i,:,4] > 0.9, :]
            pick = box_ops.nms(selected_boxes[:, :4], selected_boxes[:, 4], .5)
            selected_boxes = selected_boxes[pick,:]
            orders = selected_boxes[:, 4].argsort(descending=True)
            pred_boxes.append(selected_boxes[orders,:][:100,:].cpu())
        return pred_boxes

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int) -> torch.Tensor:

        imgs,gt_boxes = batch

        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)
        ratio = 10
        sample_factor = 5

        cls_logits,reg_logits = self(imgs)

        target_cls:List = []
        target_regs:List = []

        ignore:List = []
        rfs:List = []

        debug_fmaps = [] # ? Debug
        current = 0

        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]

            # ? Debug
            start = current
            current += (fh*fw)
            debug_fmaps.append((fh,fw,start))

            t_cls,t_regs,ig = self.heads[i].build_targets((fh,fw), gt_boxes, device=device, dtype=dtype)
            # t_cls          : bs,fh',fw'      | type: model.dtype          | device: model.device
            # t_regs         : bs,fh',fw',4    | type: model.dtype          | device: model.device
            # ig             : bs,fh',fw'      | type: torch.bool           | device: model.device

            cls_logits[i] = cls_logits[i].permute(0,2,3,1).view(batch_size, -1)
            reg_logits[i] = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)

            # TODO use cache
            h_rfs = self.heads[i].gen_rf_anchors(fh, fw, device=device, dtype=dtype, clip=True)
            rfs.append(h_rfs.view(-1,4))
            # List[ (fh*fw,4) ]

            target_cls.append(t_cls.view(batch_size, -1))
            target_regs.append(t_regs.view(batch_size,-1,4))

            ignore.append(ig.view(batch_size,-1))

        cls_logits = torch.cat(cls_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)
        target_cls = torch.cat(target_cls, dim=1)
        target_regs = torch.cat(target_regs, dim=1)
        ignore = torch.cat(ignore, dim=1)
        priors = torch.cat(rfs,dim=0)

        pos_mask = (target_cls == 1) & (~ignore)
        rpos_mask = target_cls == 1
        neg_mask = (target_cls == 0) & (~ignore)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        negatives = min(negatives,ratio*positives)

        # *Random sample selection
        ############################
        neg_mask = neg_mask.view(-1)
        ss, = torch.where(neg_mask)
        selections = random_sample_selection(ss.cpu().numpy().tolist(), negatives)
        neg_mask[:] = False
        neg_mask[selections] = True
        neg_mask = neg_mask.view(batch_size,-1)
        ############################

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[pos_mask | neg_mask], target_cls[pos_mask | neg_mask])

        reg_loss = F.mse_loss(reg_logits[rpos_mask, :], target_regs[rpos_mask, :])

        ## ? Debug
        if self.__debug:
            #debug_show_rf_matches(imgs,rfs,gt_boxes)
            debug_show_pos_neg_ig_with_gt(imgs,gt_boxes,rfs,debug_fmaps,pos_mask,neg_mask,ignore)

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
                # TODO checkout here
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
        rpos_mask = target_cls == 1
        neg_mask = (target_cls == 0) & (~ignore)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        negatives = min(negatives,ratio*positives)

        # *Random sample selection
        ############################
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

        reg_loss = F.mse_loss(reg_logits[rpos_mask, :], target_regs[rpos_mask, :])

        assert not torch.isnan(reg_loss) and not torch.isnan(cls_loss)
        loss = cls_loss + reg_loss
        pred_boxes:List = []
        for i in range(batch_size):
            selected_boxes = preds[i, preds[i,:,4] > 0.5, :]
            pick = box_ops.nms(selected_boxes[:, :4], selected_boxes[:, 4], .5)
            selected_boxes = selected_boxes[pick,:]
            orders = selected_boxes[:, 4].argsort(descending=True)
            pred_boxes.append(selected_boxes[orders,:][:200,:].cpu())

        gt_boxes = [box.cpu() for box in gt_boxes]
        return {
            'loss':loss.item(),
            'cls_loss':cls_loss.item(),
            'reg_loss':reg_loss.item(),
            'preds': pred_boxes,
            'gts': gt_boxes
        }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hyp.get('learning_rate',1e-1),
            momentum=self.hyp.get('momentum',0.9),
            weight_decay=self.hyp.get('weight_decay',0))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[600000, 1000000, 1200000, 1400000],
            gamma=0.1)

        return [optimizer], [lr_scheduler]

def debug_show_rf_matches(tensor_imgs:torch.Tensor,
        head_rfs:List[torch.Tensor], batch_gt_boxes:List[torch.Tensor]):
    imgs = debug_tensor2img(tensor_imgs)
    np_gt_boxes = [gt_boxes.cpu().long().numpy() for gt_boxes in batch_gt_boxes]
    np_head_rfs = [rfs.cpu().long().numpy() for rfs in head_rfs]
    for img,gt_boxes in zip(imgs,np_gt_boxes):
        timg = img.copy()
        for x1,y1,x2,y2 in gt_boxes:
            timg = cv2.rectangle(timg, (x1,y1),(x2,y2),(0,255,0))

        for i,rfs in enumerate(np_head_rfs):
            print(f"for head {i+1}")
            values,pick = box_ops.box_iou(torch.tensor(rfs),torch.tensor(gt_boxes)).max(dim=0)
            pick = pick.cpu().numpy().tolist()
            nimg = timg.copy()
            for j,(x1,y1,x2,y2) in enumerate(rfs[pick,:]):
                print("iou: ",values[j])
                nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2),(0,0,255),2)
            cv2.imshow("",nimg)
            cv2.waitKey(0)

def debug_show_pos_neg_ig_with_gt(tensor_imgs:torch.Tensor,
        batch_gt_boxes:List[torch.Tensor], head_rfs:List[torch.Tensor],
        debug_fmaps:List[Tuple], opos_mask:torch.Tensor,
        oneg_mask:torch.Tensor, oig_mask:torch.Tensor):

    pos_mask = opos_mask.cpu()
    neg_mask = oneg_mask.cpu()
    ig_mask = oig_mask.cpu()

    imgs = debug_tensor2img(tensor_imgs)
    np_gt_boxes = [gt_boxes.cpu().long().numpy() for gt_boxes in batch_gt_boxes]
    np_head_rfs = [rfs.cpu().long().numpy() for rfs in head_rfs]
    for img,gt_boxes,p_mask,n_mask,i_mask in zip(imgs, np_gt_boxes,
            pos_mask, neg_mask, ig_mask):
        # p_mask: 66800,
        print(f"positives: {p_mask.sum()} negatives:{n_mask.sum()} ignore:{i_mask.sum()}")
        timg = img.copy()
        for x1,y1,x2,y2 in gt_boxes:
            timg = cv2.rectangle(timg,(x1,y1),(x2,y2),(255,0,0),2)
        for (fh,fw,start),rfs in zip(debug_fmaps,np_head_rfs):
            # rfs  : 25600,4 as xmin,ymin,xmax,ymax
            crfs = ((rfs[:, [2,3]] + rfs[:, [0,1]]) // 2).astype(np.int32)
            # crfs : 25600,2 as cx,cy
            p_crfs = crfs[p_mask[start: start+fh*fw], :]
            n_crfs = crfs[n_mask[start: start+fh*fw], :]
            i_crfs = crfs[i_mask[start: start+fh*fw], :]

            for cx,cy in p_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,255,0))

            for cx,cy in n_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,0,255))

            for cx,cy in i_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,255,255))

        cv2.imshow("",timg)
        if cv2.waitKey(0) == 27:
            exit(0)

def debug_tensor2img(imgs:torch.Tensor) -> List[np.ndarray]:
    np_imgs = []
    for img in imgs:
        nimg = (img*127.5 + 127.5).permute(1,2,0).cpu().numpy().astype(np.uint8)
        #nimg = cv2.UMat(nimg).get()
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        np_imgs.append(nimg)
    return np_imgs