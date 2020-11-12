import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
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
            num_classes:int=2, num_target_regs:int=4):
        super(LFFD,self).__init__()

        if filters is None: filters = LFFD.__FILTERS__
        if rf_sizes is None: rf_sizes = LFFD.__RF_SIZES__
        if rf_strides is None: rf_strides = LFFD.__RF_STRIDES__
        if scales is None: scales = LFFD.__SCALES__
        self.num_classes = num_classes
        self.num_target_regs = num_target_regs

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
                num_classes=num_classes, num_target_regs=num_target_regs)
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

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]], batch_idx:int):

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
            start = current
            current += (fh*fw)

            debug_fmap.append((fh,fw,start))

            t_cls,t_regs,t_objness_mask,t_reg_mask,ig = self.heads[i].build_targets((fh,fw), gt_boxes, device=device, dtype=dtype)
            # t_cls          : bs,fh',fw'      | type: torch.long           | device: model.device
            # t_regs         : bs,fh',fw',4    | type: model.dtype          | device: model.device
            # t_objness_mask : bs,fh',fw'      | type: torch.bool           | device: model.device
            # t_reg_mask     : bs,fh',fw'      | type: torch.bool           | device: model.device
            # ig             : bs,fh',fw'      | type: torch.bool           | device: model.device

            cls_logits[i] = cls_logits[i].permute(0,2,3,1).view(batch_size, -1, self.num_classes)
            reg_logits[i] = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, self.num_target_regs)

            target_cls.append(t_cls.view(batch_size, -1))
            target_regs.append(t_regs.view(batch_size,-1,self.num_target_regs))

            target_objectness_mask.append(t_objness_mask.view(batch_size,-1))
            target_reg_mask.append(t_reg_mask.view(batch_size,-1))
            ignore.append(ig.view(batch_size,-1))

        cls_logits = torch.cat(cls_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)
        target_cls = torch.cat(target_cls, dim=1)
        target_regs = torch.cat(target_regs, dim=1)

        target_objectness_mask = torch.cat(target_objectness_mask, dim=1).view(-1)
        target_reg_mask = torch.cat(target_reg_mask, dim=1).view(-1)
        ignore = torch.cat(ignore, dim=1).view(-1)

        # *INFO: Shape | Device | Dtype
        #print("cls_logits: ",cls_logits.shape," device: ",cls_logits.device, " dtype: ",cls_logits.dtype)
        #print("reg_logits: ",reg_logits.shape," device: ",reg_logits.device, " dtype: ",reg_logits.dtype)
        #print("target_cls: ",target_cls.shape," device: ",target_cls.device, " dtype: ",target_cls.dtype)
        #print("target_regs: ",target_regs.shape," device: ",target_regs.device, " dtype: ",target_regs.dtype)
        #print("target_objectness_mask: ",target_objectness_mask.shape," device: ",target_objectness_mask.device, " dtype: ",target_objectness_mask.dtype)
        #print("target_reg_mask: ",target_reg_mask.shape," device: ",target_reg_mask.device, " dtype: ",target_reg_mask.dtype)
        #print("ignore: ",ignore.shape," device: ",ignore.device, " dtype: ",ignore.dtype)
        #exit(0)

        pos_mask = target_objectness_mask & ~ignore
        neg_mask = ~target_objectness_mask & ~ignore

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        negatives = min(negatives,ratio*positives)
        """
        with torch.no_grad():
            # !apply hard negative mining
            loss = F.cross_entropy(cls_logits.view(-1, self.num_classes), target_cls.view(-1) , reduction='none')
            mask = target_objectness_mask | ignore
            loss[mask] = -math.inf
            loss = loss.cpu()
            selected = loss.argsort(descending=True)[:negatives]
            neg_mask[:] = False
            neg_mask[selected] = True
        """
        ss, = torch.where(neg_mask)
        ss = random_sample_selection(ss.cpu().numpy().tolist(), negatives)
        neg_mask[:] = False
        neg_mask[ss] = True
        negatives = neg_mask.sum()

        pos_mask = pos_mask.view(batch_size,-1)
        neg_mask = neg_mask.view(batch_size,-1)

        cls_loss = F.cross_entropy(cls_logits[pos_mask | neg_mask], target_cls[pos_mask | neg_mask])

        target_reg_mask = target_reg_mask.view(batch_size,-1)
        reg_loss = F.mse_loss(reg_logits[target_reg_mask, :], target_regs[target_reg_mask, :])

        """
        ## ? Debug
        neg_counter = 0
        for i in range(batch_size):
            img = imgs[i]
            img = img * 127.5 + 127.5
            nimg = (img*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            nimg = cv2.UMat(nimg).get()
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            for x1,y1,x2,y2 in gt_boxes[i].cpu().numpy().astype(np.int32):
                nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2), (255,0,0))

            for head,(fh,fw,start) in zip(self.heads,debug_fmap):
                priors = head.gen_rf_anchors(fh,fw,clip=True).view(-1,4)
                p_mask = pos_mask[i][start: start+fh*fw]
                n_mask = neg_mask[i][start: start+fh*fw]
                pos_boxes = priors[p_mask].cpu().numpy().astype(np.int32).tolist()
                neg_boxes = priors[n_mask].cpu().numpy().astype(np.int32).tolist()
                neg_counter += len(neg_boxes)
                for x1,y1,x2,y2 in pos_boxes:
                    nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2), (0,255,0))
                for x1,y1,x2,y2 in neg_boxes:
                    nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2), (0,0,255))

            cv2.imshow("",nimg)
            if cv2.waitKey(0) == 27:
                exit(0)
        print(negatives," ==? ",neg_counter)
        """

        assert not torch.isnan(reg_loss) and not torch.isnan(cls_loss)

        """
        if (batch_idx+1) % 20 == 0:
            print("positives: ",pos_mask.sum()," negatives: ",neg_mask.sum())
            preds = F.softmax(cls_logits[pos_mask | neg_mask], dim=1)
            print(preds[:, 0], target_cls[pos_mask])
            print(preds[:, 1], target_cls[neg_mask])
            input("waiting")
        """

        return cls_loss + reg_loss

def debug(imgs, data, batch_idx):

    batch_size = imgs.size(0)
    from cv2 import cv2
    import numpy as np

    for i,img in enumerate(imgs):
        nimg = (img*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg,cv2.COLOR_RGB2BGR)
        t_img = nimg.copy()
        for d in data:
            if isinstance(d,type(None)):
                continue
            pos_selections,neg_selections,fh,fw,head_idx = d
            pos_mask = np.zeros((batch_size*fh*fw), dtype=np.uint8)
            neg_mask = np.zeros((batch_size*fh*fw), dtype=np.uint8)

            pos_mask[pos_selections.cpu().numpy()] = 255
            neg_mask[neg_selections.cpu().numpy()] = 255

            pos_mask = pos_mask.reshape(batch_size,fh,fw)
            neg_mask = neg_mask.reshape(batch_size,fh,fw)

            overlap = pos_mask[i] == neg_mask[i]
            is_same = (np.bitwise_or(pos_mask[i][overlap] == 255, neg_mask[i][overlap] == 255)).any()
            print(f"batch idx: {batch_idx} | head idx: {head_idx} overlaps {is_same} | positives: {(pos_mask[i]==255).sum()} | negatives: {(neg_mask[i]==255).sum()}")
            cv2.imshow("",t_img)
            cv2.imshow("pos mask", pos_mask[i])
            cv2.imshow("neg mask", neg_mask[i])

            if cv2.waitKey(0) == 27:
                exit(0)

if __name__ == "__main__":
    model = LFFD()
    print(model)