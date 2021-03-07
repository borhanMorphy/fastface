import torch

def jaccard_vectorized(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
    inter = intersect(box_a,box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
                (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) # [A,B]

    union = area_a + area_b - inter
    return inter / union

def intersect(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
    """
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Args:
        box_a (torch.Tensor): [description]
        box_b (torch.Tensor): [description]
    Returns:
        torch.Tensor: [description]
    """

    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def cxcywh2xyxy(o_boxes: torch.Tensor) -> torch.Tensor:
    boxes = o_boxes.clone()
    single_batch = len(boxes.shape) == 2
    boxes = boxes.unsqueeze(0) if single_batch else boxes # N,4 => 1,N,4

    w_half = boxes[:, :, 2] / 2
    h_half = boxes[:, :, 3] / 2

    boxes[:, :, 2] = boxes[:, :, 0] + w_half
    boxes[:, :, 3] = boxes[:, :, 1] + h_half
    boxes[:, :, 0] = boxes[:, :, 0] - w_half
    boxes[:, :, 1] = boxes[:, :, 1] - h_half

    return boxes.squeeze(0) if single_batch else boxes

def xyxy2cxcywh(o_boxes: torch.Tensor) -> torch.Tensor:
    boxes = o_boxes.clone()
    single_batch = len(boxes.shape) == 2
    boxes = boxes.unsqueeze(0) if single_batch else boxes # N,4 => 1,N,4

    # x1,y1,x2,y2
    w = boxes[:, :, 2] - boxes[:, :, 0]
    h = boxes[:, :, 3] - boxes[:, :, 1]

    boxes[:, :, :2] = (boxes[:, :, :2] + boxes[:, :, 2:]) / 2
    boxes[:, :, 2] = w
    boxes[:, :, 3] = h

    return boxes.squeeze(0) if single_batch else boxes