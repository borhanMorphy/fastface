__all__ = ["BinaryFocalLoss", "DIoULoss"]

# classification losses
from .focal_loss import BinaryFocalLoss
# regression losses
from .iou_loss import DIoULoss
