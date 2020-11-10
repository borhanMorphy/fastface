import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv_layer
from .resblock import ResBlock
from .detection import DetectionHead

class LFFD(nn.Module):
    # these configs are for 8 headed lffd detector
    __FILTERS__ = [64,64,64,64,128,128,128,128]
    __RF_SIZES__ = [55,71,111,143,223,383,511,639]
    __RF_STRIDES__ = [4,4,8,8,16,32,32,32]
    __SCALES__ = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)] # calculated for 640 image input

    def __init__(self, in_channels:int=3, filters:List[int]=None,
            rf_sizes:List[int]=None, rf_strides:List[int]=None, scales:List[int]=None):
        super(LFFD,self).__init__()

        if filters is None: filters = LFFD.__FILTERS__
        if rf_sizes is None: rf_sizes = LFFD.__RF_SIZES__
        if rf_strides is None: rf_strides = LFFD.__RF_STRIDES__
        if scales is None: scales = LFFD.__SCALES__

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
            DetectionHead(idx+1,num_of_filters,rf_size,rf_stride,lower_scale,upper_scale)
            for idx,(num_of_filters, rf_size, rf_stride, (lower_scale,upper_scale)) in enumerate(zip(filters,rf_sizes,rf_strides,scales))
        ])

    def forward(self, x:torch.Tensor) -> List[torch.Tensor]:
        logits:List = []
        # *tiny part
        c1 = self.downsample_conv1(x)
        c2 = self.downsample_conv2(c1)
        c4 = self.res_block1(c2)
        c6 = self.res_block2(c4)
        c8 = self.res_block3(c6)
        logits.append( self.heads[0](c8))

        c10 = self.res_block4(c8)
        logits.append( self.heads[1](c10))

        # *small part
        c11 = self.downsample_conv3(c10)
        c13 = self.res_block4(c11)
        logits.append( self.heads[2](c13))

        c15 = self.res_block5(c13)
        logits.append( self.heads[3](c15))

        # *medium part
        c16 = self.downsample_conv4(c15)
        c18 = self.res_block6(c16)
        logits.append( self.heads[4](c18))

        # *large part
        c19 = self.downsample_conv5(c18)
        c21 = self.res_block7(c19)
        logits.append( self.heads[5](c21))

        c23 = self.res_block8(c21)
        logits.append( self.heads[6](c23))

        c25 = self.res_block9(c23)
        logits.append( self.heads[7](c25))

        return logits

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]], batch_idx:int):

        imgs, gt_boxes = batch
        head_logits = self(imgs)
        losses = []
        batched_debug_data = []

        for logits,head in zip(head_logits,self.heads):
            loss,debug_data = head.compute_loss(logits, gt_boxes)
            # debug_data: (pos_selections, neg_selections, fh, fw, self.head_idx)
            losses.append(loss)
            batched_debug_data.append(debug_data)

        logs = []
        for i,loss in enumerate(losses):
            logs.append(f"head {i+1} loss: {loss}")
        #print(" |" .join(logs))
        #debug(imgs, batched_debug_data, batch_idx)
        return sum(losses)

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