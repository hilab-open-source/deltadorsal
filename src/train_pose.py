"""Main training script for 3D hand pose estimation using DeltaDorsal."""

from typing import Any, Dict, Optional, Tuple

import rootutils
import hydra
import lightning as L
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils

from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging_utils import log_hyperparameters

import signal

signal.signal(signal.SIGUSR1, signal.SIG_DFL)

torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    wandb.finish()  # closes any existing run
    logger = instantiate_loggers(cfg.get("logger"))

    trainer = pl.Trainer(
        accelerator="gpu",
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=1.0,
        precision="bf16",
        # strategy='ddp_find_unused_parameters_true',
        callbacks=[LearningRateMonitor("epoch")] + callbacks,
        benchmark=True,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
    }

    if logger:
        log_hyperparameters(object_dict, logger)

    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = cfg.get("ckpt_path")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print(f"Best ckpt path: {ckpt_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="train_pose.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
