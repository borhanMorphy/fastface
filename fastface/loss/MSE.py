import torch
import torch.nn.functional as F

class L2Loss():
    """Mean Squared Error
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dtype = input.dtype
        device = input.device
        if input.size(0) == 0:
            # pylint: disable=not-callable
            return torch.tensor(0, dtype=dtype, device=device, requires_grad=True)
        return F.mse_loss(input, target)