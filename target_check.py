import torch
import pytorch_lightning as pl
import fastface as ff
import numpy as np
from cv2 import cv2

pl.seed_everything(42)

dataset_path = "/home/morphy/datasets/FDDB"

arch = "lffd"
config = "slim"

transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=480),
    ff.transforms.Padding(target_size=(480, 480)),
    ff.transforms.FaceDiscarder(min_face_size=10),
    ff.transforms.Normalize(mean=0, std=255)
)

model = ff.FaceDetector.build(arch, config=config)

dm = ff.datamodule.FDDBDataModule(source_dir=dataset_path, batch_size=4,
    train_transforms=transforms, val_transforms=transforms)

dm.setup(stage="fit")

dl = dm.val_dataloader()

batch, raw_targets = next(iter(dl))

cls_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')#ff.loss.BinaryFocalLoss()
reg_loss_fn = torch.nn.MSELoss(reduction='none')

model.cuda()
batch = batch.cuda()

for index, target in enumerate(raw_targets):
    for k,v in target.items():
        raw_targets[index][k] = v.cuda()

check_step = 100
counter = 0
## overfit
while 1:
    counter += 1
    model.zero_grad()

    logits = model.arch.forward(batch)
    # logits: b, N, 5

    targets = model.arch.build_targets(logits, raw_targets, input_shape=batch.shape)
    # targets: b, N, 5

    cls_targets = targets[:, :, 4]
    reg_targets = targets[:, :, :4]

    cls_logits = logits[:, :, 4]
    reg_logits = logits[:, :, :4]

    pos_mask = cls_targets == 1
    neg_mask = cls_targets == 0
    num_of_positives = pos_mask.sum()
    
    pos_cls_loss = cls_loss_fn(cls_logits[pos_mask], cls_targets[pos_mask])
    neg_cls_loss = cls_loss_fn(cls_logits[neg_mask], cls_targets[neg_mask])
    order = neg_cls_loss.argsort(descending=True)

    cls_loss = torch.cat([pos_cls_loss, neg_cls_loss[order][:num_of_positives*10]]).mean()
    reg_loss = reg_loss_fn(reg_logits[pos_mask], reg_targets[pos_mask]).mean()

    loss = cls_loss + reg_loss
    loss.backward()
    print(f"cls_loss :{cls_loss}\treg_loss: {reg_loss}\tloss:{loss}")
    with torch.no_grad():
        for p in model.parameters():
            new_p = p - 0.1 * p.grad
            p.copy_(new_p)

    if counter >= check_step:
        counter = 0
        with torch.no_grad():
            preds = model.arch.postprocess(logits, input_shape=batch.shape).cpu()
            # preds: b, N, 6

        batch_ids = preds[:, 5].long().numpy()
        scores = preds[:, 4].numpy()
        boxes = preds[:, :4].long().numpy()

        for batch_idx, img in enumerate(batch):
            img = (img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            cimg = img[:, :, [2, 1, 0]].copy()

            for x1,y1,x2,y2 in boxes[batch_idx==batch_ids]:
                cimg = cv2.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow("", cimg)
            if cv2.waitKey(0) == 27:
                break
