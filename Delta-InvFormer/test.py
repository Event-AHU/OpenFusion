import os
import yaml
import pytorch_lightning as pl
from typing import List
from argparse import ArgumentParser
from train import load_config, prepare_dataloader, prepare_model

pl.seed_everything(1234)


def main():
    # args and config
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config/")
    parser.add_argument('--config_name', type=str, default="Delta_InvFormerss.yaml")
    parser.add_argument('--model_relative_path', type=str, default="/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c02/DATA/jinliye/sihao/scotch-and-soda-main/output/checkpoint/ResNet50/epoch=10-step=96723-v2.ckpt")
    args = parser.parse_args()

    dataset_config, model_config, trainer_config = load_config(args.config_path, args.config_name)

    # prepare dataloader
    testing_dataloader = prepare_dataloader(dataset_config, batch_size=trainer_config["batch_size"], mode="test")

    # prepare model
    LightningNetwork = prepare_model(model_config, is_train=False)

    # prepare testing pipeline
    model = LightningNetwork.load_from_checkpoint(
        checkpoint_path=args.model_relative_path,
        configs=model_config
    )

    # set gpu for pl_trainer
    pl_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
    )

    # test
    result = pl_trainer.test(model, dataloaders=testing_dataloader)
    print("Testing results:", result)


if __name__ == '__main__':
    main()
