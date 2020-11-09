import torch
import torch.nn as nn
from .conv import conv_layer

class ResBlock(nn.Module):
    def __init__(self, features:int):
        super(ResBlock,self).__init__()
        self.conv1 = conv_layer(features,features)
        self.conv2 = conv_layer(features,features)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)       # c(i) => c(i+1)
        x = self.conv2(x) + input   # c(i+1) => c(i+2)
        return x