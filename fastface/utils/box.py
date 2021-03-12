import torch
from typing import Tuple
from torchvision.ops import nms

def generate_grids(fh: int, fw: int) -> torch.Tensor:
    """generates grids using given feature map dimension

    Args:
        fh (int): height of the feature map
        fw (int): width of the feature map

    Returns:
        torch.Tensor: fh x fw x 2 as x1,y1
    """
    # y: fh x fw
    # x: fh x fw
    y, x = torch.meshgrid(
        torch.arange(fh),
        torch.arange(fw)
    )

    # grids: fh x fw x 2
    return torch.stack([x, y], dim=2).float()

def jaccard_vectorized(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Calculates jaccard index with a vectorized fashion

    Args:
        box_a (torch.Tensor): torch.Tensor(A,4) as xmin,ymin,xmax,ymax
        box_b (torch.Tensor): torch.Tensor(B,4) as xmin,ymin,xmax,ymax

    Returns:
        torch.Tensor: IoUs as torch.Tensor(A,B)
    """
    inter = intersect(box_a,box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
                (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) # [A,B]

    union = area_a + area_b - inter
    return inter / union

def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Calculates intersection area of boxes given
    Args:
        box_a (torch.Tensor): torch.Tensor(A,4) as xmin,ymin,xmax,ymax
        box_b (torch.Tensor): torch.Tensor(B,4) as xmin,ymin,xmax,ymax
    Returns:
        torch.Tensor: torch.Tensor(A,B)
    """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def cxcywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert box coordiates, centerx centery width height to xmin ymin xmax ymax

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as centerx centery width height

    Returns:
        torch.Tensor: torch.Tensor(N,4) as xmin ymin xmax ymax
    """
    o_boxes = boxes.unsqueeze(0)

    w_half = o_boxes[:, :, 2] / 2
    h_half = o_boxes[:, :, 3] / 2

    o_boxes[:, :, 0] = o_boxes[:, :, 0] - w_half
    o_boxes[:, :, 1] = o_boxes[:, :, 1] - h_half
    o_boxes[:, :, 2] = o_boxes[:, :, 0] + w_half
    o_boxes[:, :, 3] = o_boxes[:, :, 1] + h_half

    return o_boxes

def xyxy2cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert box coordiates, xmin ymin xmax ymax to centerx centery width height

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as xmin ymin xmax ymax

    Returns:
        torch.Tensor: torch.Tensor(N,4) as centerx centery width height
    """
    o_boxes = boxes.unsqueeze(0)

    # x1,y1,x2,y2
    w = o_boxes[:, :, 2] - o_boxes[:, :, 0]
    h = o_boxes[:, :, 3] - o_boxes[:, :, 1]

    o_boxes[:, :, :2] = (o_boxes[:, :, :2] + o_boxes[:, :, 2:]) / 2
    o_boxes[:, :, 2] = w
    o_boxes[:, :, 3] = h

    return o_boxes

@torch.jit.script
def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, batch_ids: torch.Tensor,
    iou_threshold: float = 0.4, top_k: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies batched non max suppression to given boxes

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as xmin ymin xmax ymax
        scores (torch.Tensor): torch.Tensor(N,) as score
        batch_ids (torch.Tensor): torch.LongTensor(N,) as batch idx
        iou_threshold (float, optional): nms threshold. Defaults to 0.4.
        top_k (int, optional): keep topk. Defaults to 200.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (0): predictions as torch.Tensor(N',5) xmin, ymin, xmax, ymax, score
            (1): batch_ids as torch.LongTensor(N',)
    """

    if boxes.shape[0] == 0:
        return torch.empty(0, 5, dtype=boxes.dtype, device=boxes.device), torch.empty(0, dtype=torch.long)

    max_val = torch.max(boxes)
    max_batch_id = torch.max(batch_ids)

    cboxes = boxes / max_val

    offsets = batch_ids.to(boxes.dtype) # N,

    # TODO use this
    ctop_k = top_k * (max_batch_id + 1)

    cboxes += offsets.unsqueeze(1).repeat(1,4)

    pick = nms(cboxes, scores, iou_threshold)

    preds = torch.cat([boxes[pick], scores[pick].unsqueeze(1)], dim=1)
    batch_ids = batch_ids[pick]

    return preds, batch_ids