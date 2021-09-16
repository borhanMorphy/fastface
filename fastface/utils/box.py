import torch
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
    y, x = torch.meshgrid(torch.arange(fh), torch.arange(fw))

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
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]

    union = area_a + area_b - inter
    return inter / union


def jaccard_centered(wh_a: torch.Tensor, wh_b: torch.Tensor) -> torch.Tensor:
    """Calculates jaccard index of same centered boxes
    Args:
        wh_a (torch.Tensor): torch.Tensor(A,2) as width,height
        wh_b (torch.Tensor): torch.Tensor(B,2) as width,height
    Returns:
        torch.Tensor: torch.Tensor(A,B)
    """
    inter = intersect_centered(wh_a, wh_b)
    area_a = (wh_a[:, 0] * wh_a[:, 1]).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = (wh_b[:, 0] * wh_b[:, 1]).unsqueeze(0).expand_as(inter)  # [A,B]
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
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )

    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def intersect_centered(wh_a: torch.Tensor, wh_b: torch.Tensor) -> torch.Tensor:
    """Calculates intersection of same centered boxes

    Args:
        wh_a (torch.Tensor): torch.Tensor(A,2) as width,height
        wh_b (torch.Tensor): torch.Tensor(B,2) as width,height

    Returns:
        torch.Tensor: torch.Tensor(A,B)
    """

    A = wh_a.size(0)
    B = wh_b.size(0)
    min_w = torch.min(
        wh_a[:, [0]].unsqueeze(1).expand(A, B, 2),
        wh_b[:, [0]].unsqueeze(0).expand(A, B, 2),
    )

    # [A,2] -> [A,1,2] -> [A,B,2]

    min_h = torch.min(
        wh_a[:, [1]].unsqueeze(1).expand(A, B, 2),
        wh_b[:, [1]].unsqueeze(0).expand(A, B, 2),
    )
    # [B,2] -> [1,B,2] -> [A,B,2]

    return min_w[:, :, 0] * min_h[:, :, 0]


def cxcywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert box coordiates, centerx centery width height to xmin ymin xmax ymax

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as centerx centery width height

    Returns:
        torch.Tensor: torch.Tensor(N,4) as xmin ymin xmax ymax
    """

    wh_half = boxes[:, 2:] / 2

    x1y1 = boxes[:, :2] - wh_half
    x2y2 = boxes[:, :2] + wh_half

    return torch.cat([x1y1, x2y2], dim=1)


def xyxy2cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert box coordiates, xmin ymin xmax ymax to centerx centery width height

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as xmin ymin xmax ymax

    Returns:
        torch.Tensor: torch.Tensor(N,4) as centerx centery width height
    """
    wh = boxes[:, 2:] - boxes[:, :2]
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

    return torch.cat([cxcy, wh], dim=1)


@torch.jit.script
def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    batch_ids: torch.Tensor,
    iou_threshold: float = 0.4,
) -> torch.Tensor:
    """Applies batched non max suppression to given boxes

    Args:
        boxes (torch.Tensor): torch.Tensor(N,4) as xmin ymin xmax ymax
        scores (torch.Tensor): torch.Tensor(N,) as score
        batch_ids (torch.Tensor): torch.LongTensor(N,) as batch idx
        iou_threshold (float, optional): nms threshold. Defaults to 0.4.

    Returns:
        torch.Tensor: keep mask
    """
    if boxes.size(0) == 0:
        return torch.empty((0,), dtype=torch.long)

    max_val = torch.max(boxes)

    cboxes = boxes / max_val

    offsets = batch_ids.to(boxes.dtype)  # N,

    cboxes += offsets.unsqueeze(1).repeat(1, 4)

    return nms(cboxes, scores, iou_threshold)
