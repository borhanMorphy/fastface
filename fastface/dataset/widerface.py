import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset


def _parse_annotation_file(lines: List, ranges: List) -> Tuple[List, List]:
    idx = 0
    length = len(lines)

    def parse_box(box):
        x, y, w, h = [int(b) for b in box.split(" ")[:4]]
        return x, y, x + w, y + h

    ids = []
    targets = []
    while idx < length - 1:
        img_file_name = lines[idx]
        img_idx = int(img_file_name.split("-")[0])

        bbox_count = int(lines[idx + 1])

        if bbox_count == 0:
            idx += 3

            if img_idx in ranges:
                ids.append(img_file_name)
                targets.append([])
            continue

        boxes = lines[idx + 2 : idx + 2 + bbox_count]

        boxes = list(map(parse_box, boxes))

        if img_idx in ranges:
            ids.append(img_file_name)
            targets.append(boxes)
        idx = idx + len(boxes) + 2

    return ids, targets


def _get_validation_set(root_path: str, partition: str):
    val_mat = loadmat(
        os.path.join(root_path, f"eval_tools/ground_truth/wider_{partition}_val.mat")
    )
    source_image_dir = os.path.join(root_path, "WIDER_val/images")
    ids = []
    targets = []
    total = val_mat["file_list"].shape[0]
    for i in range(total):
        event_name = str(val_mat["event_list"][i][0][0])
        rows = val_mat["face_bbx_list"][i][0].shape[0]
        for j in range(rows):
            file_name = str(val_mat["file_list"][i][0][j][0][0])
            gt_select_ids = np.squeeze(val_mat["gt_list"][i][0][j][0])
            gt_boxes = val_mat["face_bbx_list"][i][0][j][0]
            ignore = np.ones((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)

            ignore[gt_select_ids - 1] = 0
            gt_boxes[:, [2, 3]] = gt_boxes[:, [2, 3]] + gt_boxes[:, [0, 1]]
            ids.append(os.path.join(source_image_dir, event_name, file_name + ".jpg"))
            gt_boxes = np.concatenate([gt_boxes, ignore], axis=1)

            mask = np.bitwise_or(
                gt_boxes[:, 0] >= gt_boxes[:, 2], gt_boxes[:, 1] >= gt_boxes[:, 3]
            )
            gt_boxes = gt_boxes[~mask, :]

            targets.append(gt_boxes)

    return ids, targets


def _get_landmark_annotations(ann_file_path: str) -> Dict[str, np.ndarray]:
    # {root_path}/annotations/train/label.txt
    mappings = dict()
    """
    {
        <image_name>: np.ndarray(N, 5, 2)
    }
    """
    assert os.path.isfile(
        ann_file_path
    ), "landmark annotation file is missing for widerface"

    with open(ann_file_path, "r") as foo:
        annotations = foo.read().split("\n")[:-1]

    idx = None

    for ann in annotations:
        if ann.startswith("#"):
            idx = os.path.basename(ann.replace("#", "").strip())
            mappings[idx] = np.empty((0, 5, 2), dtype=np.float32)
        else:
            ann_splits = ann.split(" ")
            landmarks = np.array(
                [
                    [float(ann_splits[4]), float(ann_splits[5])],  # l1_x l1_y
                    [float(ann_splits[7]), float(ann_splits[8])],  # l2_x l2_y
                    [float(ann_splits[10]), float(ann_splits[11])],  # l3_x l3_y
                    [float(ann_splits[13]), float(ann_splits[14])],  # l4_x l4_y
                    [float(ann_splits[16]), float(ann_splits[17])],  # l5_x l5_y
                ],
                dtype=np.float32,
            ).reshape(1, 5, 2)

            if float(ann_splits[4]) < 0:
                # ignore
                landmarks[:, :, :] = -1

            mappings[idx] = np.concatenate(
                [
                    mappings[idx],
                    landmarks,
                ],
                axis=0,
            )

    return mappings


class WiderFaceDataset(BaseDataset):
    """Widerface fastface.dataset.WiderFaceDataset Instance"""

    __DATASET_NAME__ = "widerface"

    __URLS__ = {
        "widerface-train": {
            "adapter": "gdrive",
            "check": {
                "WIDER_train/images/0--Parade": "312740df0cd71f60a46867d703edd7d6"
            },
            "kwargs": {
                "file_id": "0B6eKvaijfFUDQUUwd21EckhUbWs",
                "file_name": "WIDER_train.zip",
                "extract": True,
            },
        },
        "widerface-val": {
            "adapter": "gdrive",
            "check": {"WIDER_val": "31c304a9e3b85d384f25447de1159f85"},
            "kwargs": {
                "file_id": "0B6eKvaijfFUDd3dIRmpvSk8tLUk",
                "file_name": "WIDER_val.zip",
                "extract": True,
            },
        },
        "widerface-annotations": {
            "adapter": "http",
            "check": {"wider_face_split": "46114d331b8081101ebd620fbfdafa7a"},
            "kwargs": {
                "url": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip",
                "extract": True,
            },
        },
        "widerface-eval-code": {
            "adapter": "http",
            "check": {"eval_tools": "2831a12876417f414fd6017ef1e531ec"},
            "kwargs": {
                "url": "http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip",
                "extract": True,
            },
        },
        "widerface-landmark-annotations": {
            "adapter": "gdrive",
            "check": {"landmark_annotations": "eec0580b14900d694653efece1474c8d"},
            "kwargs": {
                "file_id": "1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8",
                "file_name": "retinaface_gt_v1.1.zip",
                "sub_dir": "landmark_annotations",
                "extract": True,
            },
        },
    }

    __phases__ = ("train", "val", "test")
    __partitions__ = ("hard", "medium", "easy")
    __partition_ranges__ = (
        tuple(range(21)),
        tuple(range(21, 41)),
        tuple(range(41, 62)),
    )

    def __init__(
        self,
        source_dir: str = None,
        phase: str = 'train',
        transforms=None,
        drop_keys: List[str] = None,
        **kwargs,
    ):

        # check if download
        source_dir = self.download(source_dir)

        assert os.path.exists(
            source_dir
        ), "given source directory for fddb is not exist at {}".format(source_dir)
        assert (
            phase is None or phase in WiderFaceDataset.__phases__
        ), "given phase {} is not \
            valid, must be one of: {}".format(
            phase, WiderFaceDataset.__phases__
        )

        if phase == "train":
            source_image_dir = os.path.join(
                source_dir, f"WIDER_{phase}/images"
            )  # TODO add assertion
            annotation_path = os.path.join(
                source_dir, f"wider_face_split/wider_face_{phase}_bbx_gt.txt"
            )
            with open(annotation_path, "r") as foo:
                annotations = foo.read().split("\n")
            
            ranges = list()
            for partition_range in self.__partition_ranges__:
                ranges += list(partition_range)

            raw_ids, raw_targets = _parse_annotation_file(
                annotations,
                ranges
            )

            del annotations
            ids = []
            targets = []
            for idx, target in zip(raw_ids, raw_targets):
                if len(target) == 0:
                    continue
                target = np.array(target, dtype=np.float32)
                mask = np.bitwise_or(
                    target[:, 0] >= target[:, 2], target[:, 1] >= target[:, 3]
                )
                target = target[~mask, :]
                if len(target) == 0:
                    continue
                bboxes = target.astype(np.float32).tolist()
                targets.append(
                    dict(
                        bboxes=bboxes,
                        labels=["face"] * len(bboxes)
                    )
                )
                ids.append(os.path.join(source_image_dir, idx))

            landmark_anns = _get_landmark_annotations(
                os.path.join(source_dir, "landmark_annotations", phase, "label.txt")
            )

            for i in range(len(ids)):
                key = os.path.basename(ids[i])
                targets[i]["keypoints"] = list()
                targets[i]["keypoint_ids"] = list()
                for j, points in enumerate(landmark_anns[key].flatten().reshape(-1, 2).tolist()):
                    keypoint_id = str(j//5) + "_" + str(j % 5)
                    if points[0] == -1 or points[1] == -1:
                        targets[i]["keypoints"].append([0, 0])
                        targets[i]["keypoint_ids"].append(keypoint_id + "_ignore")
                    else:
                        targets[i]["keypoints"].append(points)
                        targets[i]["keypoint_ids"].append(keypoint_id)
        else:

            ids, e_raw_targets = _get_validation_set(source_dir, "easy")
            _, m_raw_targets = _get_validation_set(source_dir, "medium")
            _, h_raw_targets = _get_validation_set(source_dir, "hard")

            targets = []
            for (e_target, m_target, h_target) in zip(e_raw_targets, m_raw_targets, h_raw_targets):
                # make sure boxes are match
                assert (e_target[:, :4] == m_target[:, :4]).all()
                assert (m_target[:, :4] == h_target[:, :4]).all()
                bboxes = e_target[:, :4].astype(np.float32).tolist()
                e_labels = e_target[:, 4].astype(np.int32).tolist()
                m_labels = m_target[:, 4].astype(np.int32).tolist()
                h_labels = h_target[:, 4].astype(np.int32).tolist()
                labels = list()

                for e_label, m_label, h_label in zip(e_labels, m_labels, h_labels):
                    # label: 1 means ignore 0 otherwise
                    # h_label: 1 => face_ignore
                    # h_label: 0 & m_label: 1 => face_hard
                    # h_label: 0 & m_label: 0 & e_label: 1 =>face_medium
                    # h_lable: 0 & m_label: 0 & e_label: 0 => face_easy
                    label = "face_easy"
                    if e_label:
                        label = "face_medium"
                    if m_label:
                        label = "face_hard"
                    if h_label:
                        label = "face_ignore"

                    labels.append(label)

                targets.append(
                    dict(
                        bboxes=bboxes,
                        labels=labels
                    )
                )
            del e_raw_targets
            del m_raw_targets
            del h_raw_targets

        super().__init__(ids, targets, transforms=transforms, drop_keys=drop_keys, **kwargs)
