#代码阅读
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from torchvision import transforms
from dataset.data_LSDIR import MyDataset
from dataset.data_Flicker import MyDataset2
from torch.utils.data import ConcatDataset, DataLoader
from utils.common import instantiate_from_config, load_state_dict

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/train_zc_eps.yaml')
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
    [
        transforms.Resize(512), 
        transforms.CenterCrop(args.patch_size),  
        transforms.ToTensor() 
    ]
    )
        
    dataset = MyDataset(train_transforms)
    dataset2 = MyDataset2(train_transforms)

    dataset_con = ConcatDataset([dataset, dataset2])
    dataloader = DataLoader(dataset_con, num_workers=4, batch_size=args.batch_size, shuffle=True)

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
