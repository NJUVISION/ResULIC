from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from torchvision import transforms
from dataset.data_LSDIR import MyDataset
from dataset.data_Flicker import MyDataset2, MyDataset3, MyDataset4, MyDataset5, MyDataset6
from torch.utils.data import ConcatDataset, DataLoader
from utils.common import instantiate_from_config, load_state_dict

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='/workspace/test/DiffEIC/configs/train_stage2.yaml')
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    )
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    # pl.seed_everything(config.lightning.seed, workers=True)
    
    train_transforms = transforms.Compose(
            [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
        )
    
    train_transforms2 = transforms.Compose(
            [transforms.Resize(args.patch_size), transforms.ToTensor()]
        )
        
    dataset = MyDataset(train_transforms)
    dataset2 = MyDataset2(train_transforms)
    dataset3 = MyDataset3(train_transforms2)
    dataset4 = MyDataset4(train_transforms2)
    dataset5 = MyDataset4(train_transforms2)
    dataset6 = MyDataset4(train_transforms2)
    dataset_con = ConcatDataset([dataset, dataset2, dataset3, dataset4, dataset5, dataset6])
    dataloader = DataLoader(dataset_con, num_workers=4, batch_size=args.batch_size, shuffle=True,drop_last=True)

    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cuda"), strict=True)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
