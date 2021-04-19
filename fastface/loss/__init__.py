__all__ = [
    "BinaryCrossEntropyLoss",
    "BinaryFocalLoss",

    "L2Loss",
    "DIoULoss"
]

# classification losses
from .BCE import BinaryCrossEntropyLoss
from .BFL import BinaryFocalLoss

# regression losses
from .MSE import L2Loss
from .DIoU import DIoULoss