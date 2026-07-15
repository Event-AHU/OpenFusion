import os
import yaml
import pytorch_lightning as pl
from typing import List
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
# 1234 2345 3456
pl.seed_everything(1234)


def load_config(config_path: str, config_name: str) -> List[dict]:
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["train"]
    print(trainer_config['experiment_name'])

    return dataset_config, model_config, trainer_config


def prepare_dataloader(dataset_config: dict, batch_size: int = 1, mode: str = "train"):
    assert mode in ["train", "test"]

    if dataset_config["dataset_name"] == "NuclearFusion":
        from data_loader.NF_Dataset import NuclearFusion_Dataset
    else:
        raise NotImplementedError("No dataset {}".format(dataset_config["dataset_name"]))
    
    dataset = NuclearFusion_Dataset

    nuclear_fusion_dataset = dataset(mode, dataset_config)  # type: ignore

    if mode == "train":
        dataloader = DataLoader(nuclear_fusion_dataset, batch_size=batch_size, shuffle=True,num_workers=16,pin_memory=True)
    else:
        dataloader = DataLoader(nuclear_fusion_dataset, batch_size=batch_size)

    return dataloader


def prepare_model(model_config: dict, is_train=True) -> pl.LightningModule:
    if model_config["model_name"] == "Delta_InvFormer":
        from model.delta_invformer import LightningNetwork
    else:
        raise NotImplementedError("No model {}".format(model_config["model_name"]))

    lightning_model = LightningNetwork(model_config)
    # lightning_model = torch.compile(lightning_model)
    

    if (model_config["pretrained_weight"] != None) and is_train:
        model = lightning_model.load_from_checkpoint(
            checkpoint_path=model_config["pretrained_weight"],
            configs=model_config,
            strict=False
        )
    else:
        model = lightning_model
    
    return model


def prepare_pl_trainer(trainer_config: dict) -> pl.Trainer:
    # tensorboard logger and learning rate monitor
    tb_dir = os.path.join(trainer_config["output_dir"], trainer_config["tb_dirname"])
    tb_logger = pl.loggers.TensorBoardLogger(tb_dir, name=trainer_config["experiment_name"], default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(trainer_config["output_dir"], trainer_config["checkpoint"]["ckpt_dirname"], trainer_config["experiment_name"]),
        save_top_k=-1, # -1 means save all models
        save_weights_only=True
    )

    pl_trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # basic config: gpus, epochs, output_dir
        devices = trainer_config['gpus'],
        max_epochs=trainer_config["max_epochs"],
        default_root_dir=trainer_config["output_dir"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
        # gpu accelerate
        accelerator=trainer_config["accelerator"],
        strategy=trainer_config["strategy"],
    )

    return pl_trainer


def main():
    # args and config
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config/")
    parser.add_argument('--config_name', type=str, default="Delata_InvFormer_config.yaml")
    args = parser.parse_args()

    dataset_config, model_config, trainer_config = load_config(args.config_path, args.config_name)

    # prepare dataloader
    training_dataloader = prepare_dataloader(dataset_config, batch_size=trainer_config["batch_size"])

    # prepare model
    model = prepare_model(model_config)
    
    # prepare training pipeline
    pl_trainer = prepare_pl_trainer(trainer_config)

    # training
    pl_trainer.fit(model, training_dataloader)


if __name__ == '__main__':
    main()
