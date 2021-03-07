import fastface as ff
import torch
import torchvision.ops.boxes as box_ops
from cv2 import cv2
import numpy as np

def imageify(data):
    return (data * 255).permute(1,2,0).numpy().astype(np.uint8)[:,:,[2,1,0]]

arch = "yolov4"
config = {
    "img_size": 608,
    "strides": [32, 16],
    "anchors": [
        [
            [0.2149, 0.3216],
            [0.2926, 0.4361],
            [0.4484, 0.6696]
        ],
        [
            [0.0558, 0.0772],
            [0.1057, 0.1519],
            [0.1644, 0.2454],
        ]
    ],
    'head_infeatures': [512, 256],
    'neck_features': 512
}

img_size = 608
iou_threshold = 0.5

model = ff.FaceDetector.build(arch, config)

transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=img_size),
    ff.transform.Padding(target_size=(img_size,img_size), pad_value=0),
    ff.transform.Normalize(mean=0, std=255),
    ff.transform.ToTensor()
)

kwargs = {
    'batch_size': 4,
    'pin_memory': True,
    'shuffle': False,
    'num_workers': 4,
}

dm = ff.datamodule.FDDBDataModule(
    "/home/morphy/Downloads/FDDB",
    val_kwargs=kwargs,
    val_transforms=transforms,
)

dm.setup(stage="fit")

dl = dm.val_dataloader()

prior_boxes = torch.cat([
    model.arch.heads[i].det_layer.anchor_box_gen._prior_boxes.reshape(-1,4) for i in range(2)], dim=0
)

for batch,targets in dl:
    for data,target in zip(batch,targets):
        ious = box_ops.box_iou((target[:, :4]), prior_boxes)
        # N,M
        print(ious.max(dim=1)[0].mean())
        img = imageify(data)

        for x1,y1,x2,y2 in target[:, :4].long().numpy().tolist():
            img = cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)

        for x1,y1,x2,y2 in prior_boxes.long().numpy().tolist():
            img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),1)

        cv2.imshow("",img)
        if cv2.waitKey(0) == 27:
            exit(0)