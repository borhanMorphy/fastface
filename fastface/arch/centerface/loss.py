import kornia
import torch
import torch.nn as nn


class SoftBinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Soft Binary Focal loss.

    .. math::
        \text{SBFL}(\hat{y}) =
        \begin{cases}
            -\alpha (1 - \hat{y})^{\gamma} \, \text{log}(\hat{y}), & \text{if y} = 1\\
            -(1 - \alpha) (1 - y)^{\beta} \, (\hat{y})^{\gamma} \, \text{log}(1-\hat{y}),  & \text{otherwise}
        \end{cases}

    where:
       - :math:`\hat{y}` is the model's estimated probability for positive class.

    Args:
        beta: Focusing parameter for negative targets, default: 4.0.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`, default: 0.5.
        gamma: Focusing parameter :math:`\gamma >= 0`, default: 2.0.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.

    """

    def __init__(
        self,
        beta: float = 4.0,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        masked_target = target.long().float()
        loss_tmp = kornia.losses.binary_focal_loss_with_logits(
            input, masked_target, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
        neg_weights = torch.pow(torch.ones_like(loss_tmp) - target, self.beta)
        neg_weights[masked_target == 1] = 1.0
        loss_tmp *= neg_weights

        if self.reduction == "none":
            loss = loss_tmp
        elif self.reduction == "mean":
            loss = torch.mean(loss_tmp)
        elif self.reduction == "sum":
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss
