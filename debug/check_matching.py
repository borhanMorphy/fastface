import fastface as ff
import torch
import time
import numpy as np
from tqdm import tqdm
from cv2 import cv2

dataset = "widerface"
arch = "yolov4"
config = "tiny"
img_size = 416

arch_pkg = ff.utils.config.get_arch_pkg(arch)
arch_cls = ff.utils.config.get_arch_cls(arch)

anchor_heads = arch_cls.get_anchor_generators(config)

transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=img_size),
    ff.transform.Padding(target_size=(img_size, img_size), pad_value=0),
    ff.transform.ToTensor(),
    arch_pkg.Matcher(config)
)

source_dir = ff.utils.cache.get_data_cache_path(dataset)

# TODO make it parametric
ds = ff.dataset.WiderFaceDataset(source_dir, transforms=transforms,
    phase='train', partitions=["easy"])

for img, targets in ds:
    """
    targets
        heads:
            (0):
                target_objectness (torch.Tensor): nA x Gy x Gx
                ignore_objectness (torch.Tensor):  nA x Gy x Gx
                target_regs (torch.Tensor):  nA x Gy x Gx x 4
            ...
        gt_boxes:
            pass
    """
    img = img.permute(1,2,0).numpy().astype(np.uint8)[:, :, [2,1,0]]
    imgh, imgw, _ = img.shape
    gt_boxes = targets['gt_boxes']

    for target, anchor_head in zip(targets['heads'], anchor_heads):
        target_objectness = target["target_objectness"]
        ignore_objectness = target["ignore_objectness"]
        target_regs = target["target_regs"]

        pos_obj_mask = (target_objectness == 1) & (ignore_objectness == 0)
        priors = anchor_head.estimated_forward(imgh, imgw)
        print(priors[pos_obj_mask])
        for x1,y1,x2,y2 in priors[pos_obj_mask].long():
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    for x1,y1,x2,y2 in gt_boxes[:, :4].long():
        img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        cx = (x2+x1)//2
        cy = (y2+y1)//2
        img = cv2.circle(img, (cx,cy), 10, (0,255,0), 2)

    cv2.imshow("",img)
    if cv2.waitKey(0) == 27:
        break