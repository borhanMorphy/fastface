from typing import List

import torch
import torch.nn as nn


def conv_block(in_feature: int, out_feature: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_feature),
        nn.ReLU(),
    )


class FPN(nn.Module):
    def __init__(
        self,
        in_features: List[int],
        out_feature: int,
        upsample_method: str = "deconv",
        block=conv_block,
    ):
        super().__init__()
        assert len(in_features) > 1, "level of fpn should be greater than 1"

        # top layer
        self.top_layer = nn.Conv2d(
            in_features[-1], out_feature, kernel_size=1, stride=1, padding=0
        )

        # upsample layer
        if upsample_method == "deconv":
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(out_feature, out_feature, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_feature),
                nn.ReLU(),
            )
        elif upsample_method == "nearest":
            upsample_layer = nn.UpsamplingNearest2d(scale_factor=2)
        elif upsample_method == "bilinear":
            upsample_layer = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            raise AssertionError("upsample method is not supported")

        self.upsample_layer = upsample_layer

        # build top down architecture
        self.top_down_layers = nn.ModuleList(
            [
                block(in_feature, out_feature)
                for in_feature in reversed(in_features[:-1])
            ]
        )

        self.levels = len(self.top_down_layers)

    def forward(self, c_in: List[torch.Tensor]) -> torch.Tensor:
        p_out = self.top_layer(c_in[self.levels])

        for i, layer in enumerate(self.top_down_layers):
            p_out = layer(c_in[self.levels - 1 - i]) + self.upsample_layer(p_out)
        return p_out
