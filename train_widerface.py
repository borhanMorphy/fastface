import pytorch_lightning as pl
import fastface as ff
from typing import Dict
import argparse
import yaml
import os

def select_checkpoint(checkpoint_path:str, key:str='epoch', is_max:bool=True): # returns max
    ckpts = []
    paths = []
    for ckpt in os.listdir(checkpoint_path):
        if not ckpt.endswith(".ckpt"): continue

        for section in ckpt.split("-")[1:]:
            k,value = section.split("=")
            if k == key:
                paths.append(os.path.join(checkpoint_path, ckpt))
                ckpts.append(float(value))
                break
    if not is_max:
        ckpts = [-1*ckpt for ckpt in ckpts]

    selected_value = max(ckpts)
    idx = ckpts.index(selected_value)
    return paths[idx]

def load_yaml_file(yaml_path:str) -> Dict:
    with open(yaml_path,"r") as foo:
        return yaml.load(foo, Loader=yaml.FullLoader)

def parse_arguments() -> Dict:
    ap = argparse.ArgumentParser()

    ap.add_argument("--yaml", "-y", type=load_yaml_file, required=True, help="configuration yaml path for training")
    ap.add_argument("--resume", "-r", action="store_true", help="resumes training using checkpoint path")
    ap.add_argument("--ckpt-path", "-ckpt", type=str)
    ap.add_argument("--seed", "-s", type=int, help="random seed")
    args = ap.parse_args()

    args.yaml.update({'ckpt_path':args.ckpt_path})

    return args

def main(kwargs:Dict, resume:bool, seed:int):
    if seed: ff.utils.random.seed_everything(seed)

    arch = kwargs['arch']
    config = kwargs['config']
    hparams = kwargs['hparams']
    in_channels = kwargs['in_channels']

    datamodule = kwargs['datamodule']
    trainer_configs = kwargs['trainer']
    ckpt_path = kwargs['ckpt_path']

    arch_configs = ff.get_arch_config(arch, config)

    arch_pkg = ff.utils.config.get_arch_pkg(arch)

    matcher = arch_pkg.Matcher(**arch_configs)

    train_transforms = ff.transform.Compose(
        ff.transform.FaceDiscarder(min_face_scale=2),
        ff.transform.LFFDRandomSample( # TODO handle different configurations
            [
                (10,15),(15,20),(20,40),(40,70),
                (70,110),(110,250),(250,400),(400,560)
            ], target_size=(640,640)),
        ff.transform.FaceDiscarder(min_face_scale=8),
        ff.transform.RandomHorizontalFlip(p=0.5),
        ff.transform.Normalize(mean=127.5, std=127.5),
        ff.transform.ToTensor()
    )

    val_transforms = ff.transform.Compose(
        ff.transform.Interpolate(max_dim=640),
        ff.transform.Padding(target_size=(640,640), pad_value=0),
        ff.transform.Normalize(mean=127.5, std=127.5),
        ff.transform.ToTensor()
    )

    datamodule['train'].update({"collate_fn":arch_pkg.Matcher.collate_fn})
    datamodule['val'].update({"collate_fn":arch_pkg.Matcher.collate_fn})

    checkpoint_dirpath = kwargs['checkpoint']['dirpath']
    if checkpoint_dirpath is None:
        checkpoint_dirpath = ff.utils.cache.get_checkpoint_cache_path(f"{arch}_{config}")

    if resume:
        ckpt_path = select_checkpoint(checkpoint_dirpath) if ckpt_path is None else ckpt_path
        if ckpt_path in ff.list_pretrained_models():
            model = ff.FaceDetector.from_pretrained(ckpt_path)
            model.hparams.update(hparams)
            print(f"resuming training with pretrained: {ckpt_path}")
        else:
            model = ff.FaceDetector.build(arch, config, hparams=hparams, num_classes=1, in_channels=in_channels)
    else:
        model = ff.FaceDetector.build(arch, config, hparams=hparams,
            num_classes=1, in_channels=in_channels)

    print(f"using checkpoint path: {checkpoint_dirpath}")

    checkpoint_verbose = kwargs['checkpoint'].get('verbose', True)
    checkpoint_filename = kwargs['checkpoint']['filename']
    checkpoint_monitor = kwargs['checkpoint']['monitor']
    checkpoint_save_top_k = kwargs['checkpoint']['save_top_k']
    checkpoint_mode = kwargs['checkpoint']['mode']

    metric = ff.metric.get_metric_by_name("widerface_ap")
    model.add_metric("widerface_ap",metric)

    dm = ff.datamodule.WiderFaceDataModule(
        partitions=datamodule['partitions'],
        train_kwargs=datamodule['train'],
        train_transforms=train_transforms,
        train_target_transform=matcher,
        val_kwargs=datamodule['val'],
        val_target_transform=matcher,
        val_transforms=val_transforms,
        test_kwargs={'batch_size':4, 'num_workers':4, 'collate_fn':arch_pkg.Matcher.collate_fn},
        test_target_transform=matcher,
        test_transforms=val_transforms
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath= checkpoint_dirpath,
        verbose= checkpoint_verbose,
        filename=checkpoint_filename.format(
            arch=arch, config=config,
            dataset=datamodule['dataset']),

        monitor=checkpoint_monitor,
        save_top_k=checkpoint_save_top_k,
        mode=checkpoint_mode
    )

    trainer = pl.Trainer(
        default_root_dir=ff.utils.cache.get_cache_path(),
        gpus=trainer_configs.get('gpus',1),
        accumulate_grad_batches=trainer_configs.get('accumulate_grad_batches',1),
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=ckpt_path,
        max_epochs=trainer_configs.get('max_epochs', 100),
        check_val_every_n_epoch=trainer_configs.get('check_val_every_n_epoch', 1),
        precision=trainer_configs.get('precision', 32),
        gradient_clip_val=trainer_configs.get('gradient_clip_val', 0))

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.yaml, args.resume, args.seed)