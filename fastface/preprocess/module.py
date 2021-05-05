from typing import Tuple
import torch
import torch.nn as nn

from . import (
    Interpolate, DummyInterpolate,
    Pad, DummyPad,
    Normalize
)

class Preprocess(nn.Module):

    def __init__(self, mean: float = 0, std: float = 1,
            target_size: Tuple[int, int] = None,
            normalized_input: bool = True, **kwargs):
        super().__init__()
        # target_size: h, w
        # TODO check types
        # TODO pydoc
        self.register_buffer(
            "divider",
            torch.tensor(255.0) if normalized_input else torch.tensor(1.0), # pylint: disable=not-callable
            persistent=False)
            
        self.normalize = Normalize(mean=mean, std=std)

        self.interpolate = DummyInterpolate() if target_size is None else Interpolate(max_size=max(target_size))

        self.pad = DummyPad() if target_size is None else Pad(target_size=target_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies preprocess to the given input

        Args:
            x (torch.Tensor): input image as torch.FloatTensor(B x C x H x W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (0): preprocessed input as torch.FloatTensor(B x C x H' x W')
                (1): scale factor as torch.FloatTensor(1)
                (2): applied padding as torch.FloatTensor(4) pad left, pad top, pad right, pad bottom
        """
        # divides 255 if inputs are normalized
        batch = x / self.divider

        # applies (batch - mean) / std
        batch = self.normalize(batch)
        
        # applies down or up sampling, using `max_size`
        batch, scale_factor = self.interpolate(batch)

        # applies padding, using `target_size`
        batch, padding = self.pad(batch)

        return (batch, scale_factor, padding)

    @torch.jit.export
    def adjust(self, boxes: torch.Tensor,
            sf: torch.Tensor, pad: torch.Tensor,
            img_dims: Tuple[int, int]) -> torch.Tensor:
        # TODO pydoc

        # fix padding
        boxes = boxes - pad[:2].repeat(2)

        # fix scale
        boxes = boxes / (sf + 1e-16)

        # fix boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_dims[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_dims[0])

        return boxes