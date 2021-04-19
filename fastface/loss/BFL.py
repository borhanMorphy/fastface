import torch

class BinaryFocalLoss():
    """Binary Focal Loss
    """

    def __init__(self, gamma: float = 2, alpha: float = 1, **kwargs):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input: torch.Tensor(N,)
        # target: torch.Tensor(N,)

        probs = torch.sigmoid(input)
        pos_mask = target == 1

        # -alpha * (1 - p_t)**gamma * log(p_t)
        pos_loss = -self.alpha * torch.pow(1-probs[pos_mask], self.gamma) * torch.log(probs[pos_mask] + 1e-16)
        neg_loss = -self.alpha * torch.pow(probs[~pos_mask], self.gamma) * torch.log(1-probs[~pos_mask] + 1e-16)

        loss = torch.cat([pos_loss,neg_loss])

        return loss.mean()