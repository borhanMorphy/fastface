from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePreprocess(nn.Module):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).reshape(1,-1,1,1))
        self.register_buffer("std", torch.tensor(std).reshape(1,-1,1,1))

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [c,h,w] => (b x c x h x w) | scales b,1 | paddings b,4
        target_h: int = max([img.size(1) for img in batch])
        target_w: int = max([img.size(2) for img in batch])

        scales: List = []
        paddings: List = []
        batch_size: int = len(batch)

        for i in range(batch_size):
            # apply interpolation
            h: int = batch[i].size(1)
            w: int = batch[i].size(2)

            scale_factor: float = min(target_h/h, target_w/w)

            batch[i] = F.interpolate(batch[i].unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                recompute_scale_factor=False,
                align_corners=False).squeeze(0)

            scales.append(scale_factor)

            new_h: int = batch[i].size(1)
            new_w: int = batch[i].size(2)

            # apply padding
            pad_left = (target_w - new_w) // 2
            pad_right = pad_left + (target_w - new_w) % 2

            pad_top = (target_h - new_h) // 2
            pad_bottom = pad_top + (target_h - new_h) % 2

            batch[i] = F.pad(batch[i],
                (pad_left, pad_right, pad_top, pad_bottom),
                value=0)

            paddings.append([pad_left, pad_top, pad_right, pad_bottom])

        batch = torch.stack(batch, dim=0)
        scales = torch.tensor(scales, dtype=batch.dtype, device=batch.device)
        paddings = torch.tensor(paddings, dtype=batch.dtype, device=batch.device)

        # apply normalization
        batch = (batch - self.mean) / self.std

        return batch, scales, paddings

    def adjust(self, preds: torch.Tensor, scales: torch.Tensor,
            paddings: torch.Tensor) -> torch.Tensor:
        """Re-adjust predictions using scales and paddings

        Args:
            preds (torch.Tensor): torch.Tensor(B, N, 5) as xmin, ymin, xmax, ymax, score
            scales (torch.Tensor): torch.Tensor(B,)
            paddings (torch.Tensor): torch.Tensor(B,4) as pad_left, pad_top, pad_right, pad_bottom

        Returns:
            torch.Tensor: torch.Tensor(B, N, 5) as xmin, ymin, xmax, ymax, score
        """
        preds[:, :, :4] = preds[:, :, :4] - paddings[:, :2].repeat(1,2).reshape(1,1,4)
        preds[:, :, :4] = preds[:, :, :4] / scales.reshape(-1,1,1)

        return preds