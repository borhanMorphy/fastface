import pytorch_lightning as pl
import fastface as ff

pl.seed_everything(42)

# configs
dataset_path = "/home/morphy/datasets/FDDB"
ckpt_path = None

arch = "lffd"
config = "slim"

accumulate_grad_batches = 1
check_val_every_n_epoch = 50
max_epochs = 100
precision = 32
gradient_clip_val = 10
batch_size = 4

train_transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=480),
    ff.transforms.Padding(target_size=(480, 480)),
    ff.transforms.FaceDiscarder(min_face_size=10),
    ff.transforms.RandomHorizontalFlip(p=0.5),
    ff.transforms.Normalize(mean=0, std=255)
)

val_transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=480),
    ff.transforms.Padding(target_size=(480, 480)),
    ff.transforms.FaceDiscarder(min_face_size=10),
    ff.transforms.Normalize(mean=0, std=255)
)

# get datamodule
dm = ff.datamodule.FDDBDataModule(source_dir=dataset_path, batch_size=batch_size,
    train_transforms=train_transforms, val_transforms=val_transforms)

"""
for img, targets in ff.dataset.FDDBDataset(dataset_path, phase='train', transforms=train_transforms):
    pil_img = ff.utils.visualize.render_targets(img, targets)
    pil_img.show()
    if input("press `q` to exit") == "q":
        break
"""

model = ff.FaceDetector.build(arch, config=config)

model.add_metric("average_precision", ff.metric.AveragePrecision())

trainer = pl.Trainer(
    overfit_batches=4, # for debug purpose
    default_root_dir="./",#ff.utils.cache.get_cache_path(),
    gpus=1,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[],#[checkpoint_callback],
    resume_from_checkpoint=ckpt_path,
    max_epochs=max_epochs,
    check_val_every_n_epoch=check_val_every_n_epoch,
    precision=precision,
    gradient_clip_val=gradient_clip_val)

trainer.fit(model, datamodule=dm)
