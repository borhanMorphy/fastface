import torch
import torch.nn.functional as F

# TODO make it nn.Module
class L2Loss():
    """Mean Squared Error"""

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO do not get mean
        # TODO use functional
        dtype = input.dtype
        device = input.device
        if input.size(0) == 0:
            # pylint: disable=not-callable
            return torch.tensor(0, dtype=dtype, device=device, requires_grad=True)
        return F.mse_loss(input, target)