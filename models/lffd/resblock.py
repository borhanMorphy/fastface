import torch
import torch.nn as nn
from .conv import conv3x3

class ResBlock(nn.Module):
    def __init__(self, features:int):
        super(ResBlock,self).__init__()
        self.conv1 = conv3x3(features,features)
        self.relu1 = nn.ReLU6()
        self.conv2 = conv3x3(features,features)
        self.relu2 = nn.ReLU6()

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)       # c(i) => c(i+1)
        x = self.relu1(x)
        x = self.conv2(x) + input   # c(i+1) => c(i+2)
        x = self.relu2(x)
        return x