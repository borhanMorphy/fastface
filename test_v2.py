import pytorch_lightning as pl
import mypackage

def collate_fn(data):
    imgs,gt_boxes = zip(*data)
    batch = [img.unsqueeze(0) for img in imgs]
    return batch,gt_boxes

def main():
    model = mypackage.module.from_checkpoint('original_lffd_560_25L_8S.pt')
    metric = mypackage.metric.get_metric("widerface_ap")
    model.add_metric("widerface_ap",metric)

    dm = mypackage.datamodule.WiderFaceDataModule(partitions=['easy'],
        test_kwargs={'batch_size':1, 'num_workers':1, 'collate_fn':collate_fn})

    trainer = pl.Trainer(gpus=1, precision=32)

    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
