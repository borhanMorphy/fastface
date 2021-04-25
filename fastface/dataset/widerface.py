__all__ = ["WiderFaceDataset"]

import os
from typing import List, Tuple
import numpy as np
from scipy.io import loadmat

from .base import BaseDataset

def _parse_annotation_file(lines: List, ranges: List) -> Tuple[List, List]:
    idx = 0
    length = len(lines)
    def parse_box(box):
        x, y, w, h = [int(b) for b in box.split(" ")[:4]]
        return x,y,x+w,y+h

    ids = []
    targets = []
    while idx < length-1:
        img_file_name = lines[idx]
        img_idx = int(img_file_name.split("-")[0])

        bbox_count = int(lines[idx+1])

        if bbox_count == 0:
            idx += 3

            if img_idx in ranges:
                ids.append(img_file_name)
                targets.append([])
            continue

        boxes = lines[idx+2:idx+2+bbox_count]

        boxes = list(map(parse_box, boxes))

        if img_idx in ranges:
            ids.append(img_file_name)
            targets.append(boxes)
        idx = idx + len(boxes) + 2

    return ids, targets

def _get_validation_set(root_path: str, partition: str):
    val_mat = loadmat(os.path.join(root_path, f'eval_tools/ground_truth/wider_{partition}_val.mat'))
    source_image_dir = os.path.join(root_path, "WIDER_val/images")
    ids = []
    targets = []
    total = val_mat['file_list'].shape[0]
    for i in range(total):
        event_name = str(val_mat['event_list'][i][0][0])
        rows = val_mat['face_bbx_list'][i][0].shape[0]
        for j in range(rows):
            file_name = str(val_mat['file_list'][i][0][j][0][0])
            gt_select_ids = np.squeeze(val_mat['gt_list'][i][0][j][0])
            gt_boxes = val_mat['face_bbx_list'][i][0][j][0]
            ignore = np.ones((gt_boxes.shape[0],1), dtype=gt_boxes.dtype)

            ignore[gt_select_ids-1] = 0
            gt_boxes[:, [2,3]] = gt_boxes[:, [2,3]] + gt_boxes[:, [0,1]]
            ids.append(os.path.join(source_image_dir,event_name,file_name+".jpg"))
            gt_boxes = np.concatenate([gt_boxes, ignore], axis=1)

            mask = np.bitwise_or(gt_boxes[:, 0] >= gt_boxes[:, 2], gt_boxes[:, 1] >= gt_boxes[:, 3])
            gt_boxes = gt_boxes[~mask, :]

            targets.append(gt_boxes)

    return ids, targets

class WiderFaceDataset(BaseDataset):
    """Widerface fastface.dataset.BaseDataset Instance"""

    __phases__ = ("train","val")
    __partitions__ = ("hard","medium","easy")
    __partition_ranges__ = ( tuple(range(21)), tuple(range(21,41)), tuple(range(41,62)) )
    def __init__(self, source_dir: str, phase: str='train',
            partitions: List = None, transforms=None, **kwargs):

        assert phase in WiderFaceDataset.__phases__, f"given phase {phase} is not valid, must be one of: {WiderFaceDataset.__phases__}"
        if not partitions:
            partitions = WiderFaceDataset.__partitions__

        for partition in partitions:
            assert partition in WiderFaceDataset.__partitions__, f"given partition {partition} is not in the defined list: {self.__partitions__}"

        if phase == 'train':
            ranges = []
            for partition in partitions:
                ranges += WiderFaceDataset.__partition_ranges__[WiderFaceDataset.__partitions__.index(partition)]
            source_image_dir = os.path.join(source_dir, f"WIDER_{phase}/images") # TODO add assertion
            annotation_path = os.path.join(source_dir, f"wider_face_split/wider_face_{phase}_bbx_gt.txt")
            with open(annotation_path,"r") as foo:
                annotations = foo.read().split("\n")
            raw_ids, raw_targets = _parse_annotation_file(annotations, ranges)
            del annotations
            ids = []
            targets = []
            for idx, target in zip(raw_ids, raw_targets):
                if len(target) == 0:
                    continue
                target = np.array(target, dtype=np.float32)
                mask = np.bitwise_or(target[:, 0] >= target[:, 2], target[:, 1] >= target[:, 3])
                target = target[~mask, :]
                if len(target) == 0:
                    continue
                targets.append({
                    "target_boxes": target
                })
                ids.append(os.path.join(source_image_dir, idx))
        else:
            # TODO each targets must be dict and handle hard parameter
            ids, raw_targets = _get_validation_set(source_dir, partitions[0])
            targets = []
            for target in raw_targets:
                targets.append({
                    "target_boxes": target[:, :4],
                    "meta": {
                        "ignore": target[:, 4]
                    }
                })
            del raw_targets

        super().__init__(ids, targets, transforms=transforms, **kwargs)