import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        dtype = input.dtype
        device = input.device
        if input.size(0) == 0:
            return torch.tensor([[0 for _ in range(target.size(-1))]], dtype=dtype, device=device, requires_grad=True)
        return F.mse_loss(input, target, reduction='none')