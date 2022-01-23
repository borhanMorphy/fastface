from typing import Tuple

import torch
import torchvision


class MobileNetV2(torchvision.models.mobilenet.MobileNetV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        delattr(self, "classifier")
        delattr(self, "last_channel")

        self.feature_layers_1 = torch.nn.Sequential(*self.features[:4])
        self.feature_layers_2 = torch.nn.Sequential(*self.features[4:7])
        self.feature_layers_3 = torch.nn.Sequential(*self.features[7:14])
        self.feature_layers_4 = torch.nn.Sequential(*self.features[14:-1])

        delattr(self, "features")

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.feature_layers_1(x)
        f2 = self.feature_layers_2(f1)
        f3 = self.feature_layers_3(f2)
        f4 = self.feature_layers_4(f3)
        return (f1, f2, f3, f4)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.hub.load_state_dict_from_url(
            torchvision.models.mobilenetv2.model_urls["mobilenet_v2"], progress=True
        )
        model.load_state_dict(state_dict, strict=False)
        return model
