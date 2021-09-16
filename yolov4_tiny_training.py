import os
import fastface as ff
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

# set seed
pl.seed_everything(41)

# build pl.LightningModule with random weights
#model = ff.FaceDetector.build_from_yaml("config_zoo/yolov4_tiny.yaml")
model = ff.FaceDetector.from_checkpoint("checkpoints/yolov4_tiny_widerface_best.ckpt")

hparams = model.hparams["hparams"]

augmentations = hparams.get("augmentations", {})

# build training transforms
train_transforms = ff.transforms.Compose(
	ff.transforms.RandomRotate(p=0.5, degree_range=augmentations.get("degree_range", 0)),
	ff.transforms.Interpolate(target_size=hparams["image_size"]),
	ff.transforms.Padding(target_size=(hparams["image_size"], hparams["image_size"])),
	ff.transforms.FaceDiscarder(min_face_size=1),
	ff.transforms.RandomHorizontalFlip(p=0.5),
    ff.transforms.RandomGaussianBlur(p=0.3, kernel_size=7, sigma=3),
	ff.transforms.ColorJitter(p=0.5,
		brightness=augmentations.get("brightness", 0),
		contrast=augmentations.get("contrast", 0),
		saturation=augmentations.get("saturation", 0)
	)
)

# build val transforms
val_transforms = ff.transforms.Compose(
	ff.transforms.Interpolate(target_size=hparams["image_size"]),
	ff.transforms.Padding(target_size=(hparams["image_size"], hparams["image_size"])),
	ff.transforms.FaceDiscarder(min_face_size=1)
)

# build torch.utils.data.DataLoader for training
train_ds = ff.dataset.WiderFaceDataset(phase="train", transforms=train_transforms)
train_dl = train_ds.get_dataloader(
	batch_size=hparams["mini_batch_size"], shuffle=True, num_workers=8
)

# build torch.utils.data.DataLoader for validation
val_ds = ff.dataset.WiderFaceDataset(phase="val", partitions=["easy"], transforms=val_transforms)
val_dl = val_ds.get_dataloader(
	batch_size=hparams["mini_batch_size"], shuffle=False, num_workers=8)

# add average precision pl.metrics.Metric to the model
model.add_metric("wAP@0.5", ff.metric.WiderFaceAP(iou_threshold=0.5))
model.add_metric("AR@0.5", ff.metric.AverageRecall(iou_threshold_min=0.5, iou_threshold_max=0.5))
model.add_metric("APsmall@0.5", ff.metric.AveragePrecision(iou_threshold=0.5, area="small"))
model.add_metric("APmedium@0.5", ff.metric.AveragePrecision(iou_threshold=0.5, area="medium"))
model.add_metric("APlarge@0.5", ff.metric.AveragePrecision(iou_threshold=0.5, area="large"))

model_save_name = "yolov4_tiny_widerface_best"
ckpt_save_path = "./checkpoints"

# resume with checkpoint, if exists
ckpt_resume_path = os.path.join(ckpt_save_path, model_save_name+".ckpt")
if not os.path.isfile(ckpt_resume_path):
	ckpt_resume_path = None

# define checkpoint callback
checkpoint_callback = ModelCheckpoint(
	dirpath=ckpt_save_path,
	verbose=True,
	filename=model_save_name,
	monitor="metrics/wAP@0.5",
	save_top_k=1,
	mode="max" # only pick max of `wAP@0.5`
)

# define lr monitor callback
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# define pl.Trainer
trainer = pl.Trainer(
	default_root_dir=".",
	accumulate_grad_batches=hparams["batch_size"] // hparams["mini_batch_size"],
	callbacks=[checkpoint_callback, lr_monitor],
	gpus=1 if torch.cuda.is_available() else 0,
	precision=32,
	#resume_from_checkpoint=ckpt_resume_path,
	max_epochs=60,
	check_val_every_n_epoch=1,
	gradient_clip_val=hparams["gradient_clip_val"]
)

# start training
trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)


""" GOAL;
{
	'AP@0.5': 0.45014679431915283,
	'APlarge@0.5': 0.9462472200393677,
	'APmedium@0.5': 0.9276220798492432,
	'APsmall@0.5': 0.3695606291294098,
	'AR@0.5': 0.626763105392456,
	'wAP@0.5': 0.8912476941196701
}
"""