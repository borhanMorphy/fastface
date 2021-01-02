import torch
import pytorch_lightning as pl
import mypackage
import time

# TODO add argparse

def collate_fn(data):
    imgs,gt_boxes = zip(*data)
    batch = [img.unsqueeze(0) for img in imgs]
    return batch,gt_boxes

def main():
    model = mypackage.module.from_pretrained(model='original_lffd_560_25L_8S')
    metric = mypackage.metric.get_metric("widerface_ap")
    model.add_metric("widerface_ap",metric)

    dm = mypackage.datamodule.WiderFaceDataModule(partitions=['easy'],
        test_kwargs={'batch_size':1, 'num_workers':1, 'collate_fn':collate_fn})

    # TODO precision 16 is much more slower
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=32)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
