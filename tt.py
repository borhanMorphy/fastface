import numpy as np
from cv2 import cv2
import torch
import fastface as ff

model = ff.FaceDetector.build("yolov4", config="tiny")

ds = ff.dataset.FDDBDataset(
    transforms=ff.transforms.Compose(
        ff.transforms.Interpolate(target_size=416),
        ff.transforms.Padding(target_size=(416, 416)),
    )
)

fmaps = [(13,13,3), (26,26,3)]

for img, target in ds:
    img = img[:, :, [2, 1, 0]]

    targets = [{
        "target_boxes": torch.from_numpy(target["target_boxes"])
    }]

    for x1,y1,x2,y2 in targets[0]["target_boxes"].long():
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0))

    n_targets, reg_targets = model.arch.build_targets(fmaps, targets)
    for head, (fh,fw,_), n_target, reg_target in zip(model.arch.heads, fmaps, n_targets, reg_targets):
        priors = head.anchor.forward(fh, fw).permute(1,2,0,3)
        pos_mask = n_target[0, :, :, :] == 1
        ignore_mask = n_target[0, :, :] == -1
        neg_mask = n_target[0, :, :] == 0
        """
        for cx,cy,w,h in priors[neg_mask]:
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255))
        """
        print(reg_target[0][pos_mask])
        for cx,cy,w,h in priors[pos_mask]:
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0))

        for cx,cy,w,h in priors[ignore_mask]:
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255))

    cv2.imshow("", img)
    if cv2.waitKey(0) == 27:
        break