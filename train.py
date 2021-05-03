import pytorch_lightning as pl
import fastface as ff

pl.seed_everything(42)

# configs
dataset_path = "/home/morphy/datasets/FDDB"
ckpt_path = None

arch = "lffd"
config = "slim"

accumulate_grad_batches = 2
check_val_every_n_epoch = 2
max_epochs = 100
precision = 32
gradient_clip_val = 10
batch_size = 16
num_workers = 8
pin_memory = True
img_size = 480
min_face_size = 10
norm_mean = 127.5
norm_std = 127.5

train_transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=img_size),
    ff.transforms.Padding(target_size=(img_size, img_size)),
    ff.transforms.FaceDiscarder(min_face_size=min_face_size),
    ff.transforms.RandomHorizontalFlip(p=0.5),
    ff.transforms.Normalize(mean=norm_mean, std=norm_std)
)

val_transforms = ff.transforms.Compose(
    ff.transforms.Interpolate(max_dim=img_size),
    ff.transforms.Padding(target_size=(img_size, img_size)),
    ff.transforms.FaceDiscarder(min_face_size=min_face_size),
    ff.transforms.Normalize(mean=norm_mean, std=norm_std)
)

# get datamodule
dm = ff.datamodule.FDDBDataModule(source_dir=dataset_path, batch_size=batch_size,
    num_workers=num_workers, pin_memory=pin_memory,
    train_transforms=train_transforms, val_transforms=val_transforms)

model = ff.FaceDetector.build(arch, config=config)

model.add_metric("average_precision", ff.metric.AveragePrecision())

trainer = pl.Trainer(
    default_root_dir="./",#ff.utils.cache.get_cache_dir(),
    gpus=1,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[],#[checkpoint_callback],
    resume_from_checkpoint=ckpt_path,
    max_epochs=max_epochs,
    check_val_every_n_epoch=check_val_every_n_epoch,
    precision=precision,
    gradient_clip_val=gradient_clip_val)

trainer.fit(model, datamodule=dm)
