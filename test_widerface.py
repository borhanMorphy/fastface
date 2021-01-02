import torch
import pytorch_lightning as pl
import fastface
import argparse

def parse_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", "-m", type=str, default="original_lffd_560_25L_8S",
        choices=fastface.list_pretrained_models(), help='pretrained model to be used')

    ap.add_argument("--device", "-d", type=str, choices=['cpu','cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu')

    ap.add_argument('--partition', '-p', type=str, default='easy',
        help='widerface partition set', choices=['easy','medium','hard'])

    return ap.parse_args()

def collate_fn(data):
    imgs,gt_boxes = zip(*data)
    batch = [img.unsqueeze(0) for img in imgs]
    return batch,gt_boxes

def main(model:str, device:str, partition:str,
        batch_size:int=1, num_workers:int=4):
    model = fastface.module.from_pretrained(model=model)

    model.summarize()

    metric = fastface.metric.get_metric("widerface_ap")
    model.add_metric("widerface_ap",metric)

    dm = fastface.datamodule.WiderFaceDataModule(partitions=[partition],
        test_kwargs={'batch_size':batch_size, 'num_workers':num_workers, 'collate_fn':collate_fn})

    # TODO precision 16 is much more slower
    trainer = pl.Trainer(
        gpus=1 if device == 'cuda' else 0,
        precision=32)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.device, args.partition)
