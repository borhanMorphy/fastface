import torch.nn as nn


def conv3x3(
    in_channels: int, out_channels: int, stride: int = 1, padding: int = 1
) -> nn.Module:

    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=True,
    )

    nn.init.xavier_normal_(conv.weight)

    if conv.bias is not None:
        conv.bias.data.fill_(0)

    return conv


def conv1x1(in_channels: int, out_channels: int) -> nn.Module:

    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
    )

    nn.init.xavier_normal_(conv.weight)

    if conv.bias is not None:
        conv.bias.data.fill_(0)

    return conv
