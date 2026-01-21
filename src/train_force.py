"""Main training script for force estimation using DeltaDorsal."""

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


@hydra.main(version_base=None, config_path="../configs", config_name="train_force.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
