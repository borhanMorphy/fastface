import torch
import pytorch_lightning as pl
import fastface as ff
import numpy as np
from cv2 import cv2

pl.seed_everything(42)

dataset_path = "/mnt/a212519a-146d-42a9-b84a-32d6e9875a58/datasets/face_detection_datasets/FDDB"

arch = "lffd"
config = "slim"

transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=640),
    ff.transforms.Padding(target_size=(640, 640)),
    ff.transforms.FaceDiscarder(min_face_size=5),
    ff.transforms.Normalize(mean=0, std=255)
)

model = ff.FaceDetector.build(arch, config=config)

dm = ff.datamodule.FDDBDataModule(source_dir=dataset_path, batch_size=1,
    train_transforms=transforms, val_transforms=transforms)

dm.setup(stage="fit")

dl = dm.val_dataloader()

batch, raw_targets = next(iter(dl))

logits = model.arch.forward(batch)
# logits: b, 5, N

preds = model.arch.postprocess(logits, input_shape=batch.shape)

targets = model.arch.build_targets(logits, raw_targets, input_shape=batch.shape)
# targets: b, 5, N

cls_targets = targets[:, [4], :].permute(0, 2, 1)
reg_targets = targets[:, :4, :].permute(0, 2, 1)

cls_logits = logits[:, [4], :].permute(0, 2, 1)
reg_logits = logits[:, :4, :].permute(0, 2, 1)

image_h, image_w = batch.shape[2:]

counter = 0

img = (batch[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

for head in model.arch.heads:
    cimg = img[:, :, [2, 1, 0]].copy()
    fh = image_h // head.anchor.rf_stride - 1
    fw = image_w // head.anchor.rf_stride - 1
    start_index = counter
    end_index = start_index + fh*fw
    counter += (fh*fw)

    head_cls_targets = cls_targets[:, start_index:end_index, :].view(-1, fh, fw, 1)
    head_reg_targets = reg_targets[:, start_index:end_index, :].view(-1, fh, fw, 4)

    head_cls_logits = cls_logits[:, start_index:end_index, :].view(-1, fh, fw, 1)
    head_reg_logits = reg_logits[:, start_index:end_index, :].view(-1, fh, fw, 4)

    rfs = head.anchor.forward(fh, fw)
    rf_centers = (rfs[..., [2, 3]] + rfs[..., [0, 1]]) / 2

    selected_fh, selected_fw = torch.where(head_cls_targets == 1)[1:3]
    print(rf_centers[selected_fh, selected_fw, :].long())

    for cx, cy in rf_centers[selected_fh, selected_fw, :].long():
        cimg = cv2.circle(cimg, (cx, cy), 5, (0, 255, 0))

    cv2.imshow("", cimg)
    if cv2.waitKey(0) == 27:
        break
