import pytorch_lightning as pl
import fastface as ff
import torch

ff.utils.random.seed_everything(42)

arch = "lffd"
config = "original"

arch_configs = ff.get_arch_config(arch, config)

arch_pkg = ff.utils.config.get_arch_pkg(arch)

matcher = arch_pkg.Matcher(**arch_configs)

train_transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=416),
    ff.transform.Padding(target_size=(416,416), pad_value=0),
    ff.transform.Normalize(mean=0, std=255),
    ff.transform.ToTensor()
)

val_transforms = ff.transform.Compose(
    ff.transform.Interpolate(max_dim=416),
    ff.transform.Padding(target_size=(416,416), pad_value=0),
    ff.transform.Normalize(mean=0, std=255),
    ff.transform.ToTensor()
)

model = ff.FaceDetector.build("lffd", "original")

metric = ff.metric.get_metric_by_name("widerface_ap")
model.add_metric("widerface_ap", metric)

train_kwargs = {
    'batch_size': 16,
    'pin_memory': True,
    'shuffle': True,
    'num_workers': 8,
    'collate_fn': matcher.collate_fn
}

val_kwargs = {
    'batch_size': 4,
    'pin_memory': True,
    'shuffle': False,
    'num_workers': 4,
    'collate_fn': matcher.collate_fn
}

dm = ff.datamodule.WiderFaceDataModule(
    partitions=["easy"],
    train_kwargs=train_kwargs,
    train_transforms=train_transforms,
    train_target_transform=matcher,
    val_kwargs=val_kwargs,
    val_target_transform=matcher,
    val_transforms=val_transforms
)

checkpoint_dirpath = ff.utils.cache.get_checkpoint_cache_path(f"{arch}_{config}")
checkpoint_filename = "{arch}_{config}_{dataset}-{{epoch:02d}}-{{val_loss:.3f}}-{{widerface_ap:.2f}}"
checkpoint_monitor = "val_loss"
checkpoint_save_top_k = 3
checkpoint_mode = 'min'
ckpt_path = None

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = checkpoint_dirpath,
    verbose = True,
    filename=checkpoint_filename.format(
        arch=arch, config=config,
        dataset="widerface"),

    monitor=checkpoint_monitor,
    save_top_k=checkpoint_save_top_k,
    mode=checkpoint_mode
)

trainer = pl.Trainer(
    default_root_dir=ff.utils.cache.get_cache_path(),
    gpus=1,
    accumulate_grad_batches=2,
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=ckpt_path,
    max_epochs=100,
    check_val_every_n_epoch=1,
    precision=32,
    gradient_clip_val=0)
print(model.mean, model.std)
trainer.model = model
trainer.save_checkpoint("/home/morphy/.cache/fastface/0.1.0rc1/model/lffd_original.ckpt");exit(0)
trainer.fit(model, datamodule=dm)