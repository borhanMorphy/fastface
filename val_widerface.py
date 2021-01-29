# TODO remove this script after resolving training for lffd
import torch
import pytorch_lightning as pl
import fastface as ff
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", "-m", type=str, default="original_lffd_560_25L_8S",
        choices=ff.list_pretrained_models(), help='pretrained model to be used')

    ap.add_argument("--device", "-d", type=str, choices=['cpu','cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu')

    ap.add_argument('--partition', '-p', type=str, default='easy',
        help='widerface partition set', choices=['easy','medium','hard'])

    return ap.parse_args()

def main(model:str, device:str, partition:str,
        batch_size:int=1, num_workers:int=4):
    model = ff.module.from_pretrained(model=model)

    model.summarize()

    arch_configs = ff.get_arch_config("lffd", "560_25L_8S")

    arch_pkg = ff.utils.config.get_arch_pkg("lffd")

    matcher = arch_pkg.Matcher(**arch_configs)

    transforms = ff.transform.Compose(
        ff.transform.Interpolate(max_dim=640),
        ff.transform.Padding(target_size=(640,640), pad_value=0),
        ff.transform.Normalize(mean=127.5, std=127.5),
        ff.transform.ToTensor()
    )

    dm = ff.datamodule.WiderFaceDataModule(
            partitions=["easy"],
            train_kwargs={'batch_size':4, 'num_workers':4, 'collate_fn':arch_pkg.Matcher.collate_fn},
            train_target_transform=matcher,
            val_kwargs={'batch_size':4, 'num_workers':4, 'collate_fn':arch_pkg.Matcher.collate_fn},
            val_target_transform=matcher,
            test_kwargs={'batch_size':4, 'num_workers':4, 'collate_fn':arch_pkg.Matcher.collate_fn},
            test_target_transform=matcher,
            test_transforms=transforms
        )

    metric = ff.metric.get_metric("widerface_ap")
    model.add_metric("widerface_ap",metric)

    # TODO precision 16 is much more slower
    trainer = pl.Trainer(
        gpus=1 if device == 'cuda' else 0,
        precision=32)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.device, args.partition)
