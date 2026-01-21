"""PyTorch Lightning module for force estimation training."""

import os
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl

import torchmetrics as tm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

import wandb
import hydra
import time


class DeltaDorsalForceModule(pl.LightningModule):
    def __init__(
        self,
        class_counts,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        unfreeze=[],
        lambda_p: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.backbone = backbone
        self.head = head

        counts = np.array(class_counts, dtype=float)
        freqs = counts / counts.sum()
        class_weights = torch.tensor(1.0 / (freqs + 1e-12), dtype=torch.float)

        self.force_loss = nn.CrossEntropyLoss(weight=class_weights)

        K = self.head.n_classes
        metrics = {
            "acc": MulticlassAccuracy(num_classes=K, average="micro"),
            "prec_w": MulticlassPrecision(num_classes=K, average="weighted"),
            "recall_w": MulticlassRecall(num_classes=K, average="weighted"),
            "f1_w": MulticlassF1Score(num_classes=K, average="weighted"),
        }
        self.train_metrics = tm.MetricCollection(metrics)
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()

    def get_parameters(self):
        all_params = list(self.head.parameters())
        all_params += list(self.backbone.parameters())

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, b):
        feats = self.backbone(b["img"], b["base_img"])
        logits, probs = self.head(feats)

        preds = probs.argmax(dim=-1)

        return logits, probs, preds

    def model_step(self, b):
        B = b["img"].shape[0]
        logits, probs, preds = self.forward(b)

        targets = b["tap_label"].long()

        loss = self.force_loss(
            logits.reshape(B * self.head.out_dim, self.head.n_classes),
            targets.reshape(B * self.head.out_dim),
        )

        return loss, preds, targets

    def training_step(self, b, b_idx):
        loss, preds, targets = self.model_step(b)

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def on_validation_start(self):
        self._val_outputs = {"pred": [], "target": []}

    def validation_step(self, b, b_idx):
        B = b["img"].shape[0]

        start_time = time.time()
        loss, preds, targets = self.model_step(b)
        elapsed_time = int((time.time() - start_time) * 1000)

        self._val_outputs["pred"].append(preds)
        self._val_outputs["target"].append(targets)

        self.val_metrics.update(preds.view(-1), targets.view(-1))

        m = self.val_metrics.compute()
        self.log_dict(
            {f"val/{k}": v for k, v in m.items()},
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/inference-time",
            elapsed_time / B,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_end(self):
        self.val_metrics.reset()

        for key, val in self._val_outputs.items():
            self._val_outputs[key] = (
                torch.concat(val).reshape(-1, self.head.out_dim).detach().cpu().numpy()
            )

        for i in range(self.head.out_dim):
            self.logger.experiment.log(
                {
                    f"val/conf_mat_{i}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=self._val_outputs["target"][:, i],
                        preds=self._val_outputs["pred"][:, i],
                    )
                }
            )

        for key, val in self._val_outputs.items():
            self._val_outputs[key] = []

    def on_test_start(self):
        self._test_outputs = {
            "frame_id": [],
            "pred_tap_labels": [],
            "gt_tap_labels": [],
        }

    def test_step(self, b, b_idx):
        B = b["img"].shape[0]

        start_time = time.time()
        loss, preds, targets = self.model_step(b)
        elapsed_time = int((time.time() - start_time) * 1000)

        self.test_metrics.update(preds.view(-1), targets.view(-1))

        self._test_outputs["frame_id"].append(b["frame_id"])
        self._test_outputs["pred_tap_labels"].append(preds)
        self._test_outputs["gt_tap_labels"].append(targets)

        m = self.test_metrics.compute()
        self.log_dict(
            {f"test/{k}": v for k, v in m.items()},
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/inference-time",
            elapsed_time / B,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        self.test_metrics.reset()

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(
            output_dir, f"test_outputs_epoch{self.current_epoch}.npy"
        )

        final_outputs = {}
        for key, val in self._test_outputs.items():
            local = torch.concat(val)
            gathered = self.all_gather(local)
            final_outputs[key] = gathered.detach().cpu().to(torch.int).numpy()

        final_outputs["pred_tap_labels"] = final_outputs["pred_tap_labels"].reshape(
            -1, self.head.out_dim
        )
        final_outputs["gt_tap_labels"] = final_outputs["gt_tap_labels"].reshape(
            -1, self.head.out_dim
        )

        np.save(file_path, final_outputs, allow_pickle=True)

        for i in range(self.head.out_dim):
            self.logger.experiment.log(
                {
                    f"test/conf_mat_{i}": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=final_outputs["gt_tap_labels"][:, i],
                        preds=final_outputs["pred_tap_labels"][:, i],
                    )
                }
            )

        for key, val in self._test_outputs.items():
            self._test_outputs[key] = []

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
