from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(
    images: Union[np.ndarray, List], dtype: str, device: str
) -> List[torch.Tensor]:
    """Converts given image or list of images to list of tensors
    Args:
        images (Union[np.ndarray, List]): RGB image or list of RGB images
        dtype (str): target dtype as string
        device (str): target device as string
    Returns:
        List[torch.Tensor]: list of torch.Tensor(C x H x W)
    """
    assert isinstance(
        images, (list, np.ndarray)
    ), "give images must be eather list of numpy arrays or numpy array"

    if isinstance(images, np.ndarray):
        images = [images]

    batch: List[torch.Tensor] = []

    for img in images:
        assert (
            len(img.shape) == 3
        ), "image shape must be channel, height\
            , with length of 3 but found {}".format(
            len(img.shape)
        )
        assert (
            img.shape[2] == 3
        ), "channel size of the image must be 3 but found {}".format(img.shape[2])

        batch.append(
            # h,w,c => c,h,w
            # pylint: disable=not-callable
            torch.tensor(img, dtype=dtype, device=device).permute(2, 0, 1)
        )

    return batch


def to_json(preds: List[torch.Tensor]) -> List[Dict]:
    """Converts given list of tensor predictions to json serializable format
    Args:
        preds (List[torch.Tensor]): list of torch.Tensor(N,5) as xmin, ymin, xmax, ymax, score
    Returns:
        List[Dict]: [
            # single image results
            {
                    "boxes": <array>,  # List[List[xmin, ymin, xmax, ymax]]
                    "scores": <array>  # List[float]
            },
            ...
        ]
    """
    results: List[Dict] = []

    for pred in preds:
        if pred.size(0) != 0:
            pred = pred.cpu().numpy()
            boxes = pred[:, :4].astype(np.int32).tolist()
            scores = pred[:, 4].tolist()
        else:
            boxes = []
            scores = []

        results.append({"boxes": boxes, "scores": scores})

    return results


def prepare_batch(
    batch: List[torch.Tensor], target_size: int, adaptive_batch: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list of tensors to tensors

    Args:
        batch (List[torch.Tensor]): list of tensors(float) as (C x H x W)
        target_size (int): maximum dimension size to fit
        adaptive_batch (bool, optional): if true than batching will be adaptive,
                using max dimension of the batch, otherwise it will use `target_size`, Default: True

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (0) : batched inputs as B x C x target_size x target_size
            (1) : applied scale factors for each image as torch.FloatTensor(B,)
            (2) : applied padding for each image as torch.LongTensor(B,4) pad left, top, right, bottom
    """
    if adaptive_batch:
        # select max dimension in inputs
        target_size: int = max([max(img.size(1), img.size(2)) for img in batch])

    scales: List = []
    paddings: List = []

    for i, img in enumerate(batch):
        # apply interpolation
        img_h: int = img.size(-2)
        img_w: int = img.size(-1)

        scale_factor: float = min(target_size / img_h, target_size / img_w)

        img = F.interpolate(
            img.unsqueeze(0),
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        new_h: int = img.size(-2)
        new_w: int = img.size(-1)

        # apply padding
        pad_left = (target_size - new_w) // 2
        pad_right = pad_left + (target_size - new_w) % 2

        pad_top = (target_size - new_h) // 2
        pad_bottom = pad_top + (target_size - new_h) % 2

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        paddings.append([pad_left, pad_top, pad_right, pad_bottom])
        scales.append(scale_factor)
        batch[i] = img

    batch = torch.cat(batch, dim=0).contiguous()
    # pylint: disable=not-callable
    scales = torch.tensor(scales, dtype=batch.dtype, device=batch.device)
    # pylint: disable=not-callable
    paddings = torch.tensor(paddings, dtype=batch.dtype, device=batch.device)

    return batch, scales, paddings


def adjust_results(
    preds: List[torch.Tensor], scales: torch.Tensor, paddings: torch.Tensor
) -> torch.Tensor:
    """Re-adjust predictions using scales and paddings

    Args:
        preds (List[torch.Tensor]): list of torch.Tensor(N, 5) as xmin, ymin, xmax, ymax, score
        scales (torch.Tensor): torch.Tensor(B,)
        paddings (torch.Tensor): torch.Tensor(B,4) as pad_left, pad_top, pad_right, pad_bottom

    Returns:
        torch.Tensor: torch.Tensor(B, N, 5) as xmin, ymin, xmax, ymax, score
    """
    for i, pred in enumerate(preds):
        if pred.size(0) == 0:
            continue

        preds[i][:, :4] = pred[:, :4] - paddings[i, :2].repeat(1, 2)
        preds[i][:, :4] = pred[:, :4] / scales[i]

    return preds
