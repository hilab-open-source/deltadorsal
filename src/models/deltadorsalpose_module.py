"""PyTorch Lightning module for 3D hand pose estimation training."""

import os
import numpy as np

import torch

import pytorch_lightning as pl

import wandb
import hydra
import time

from models.mano_wrapper import MANOPoseOnly
from models.losses import Keypoint3DLoss, ParameterLoss
from utils.mano_utils import mano_to_openpose, calc_mpjpe
from utils.geom_utils import axis_angle_to_matrix
from utils.visualize import log_examples


class DeltaDorsalPoseModule(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        mano_path: str,
        loss_weights,
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

        self.mano_fit = MANOPoseOnly(
            model_path=mano_path, is_rhand=True, use_pca=False, flat_hand_mean=True
        )
        self.mano_fit.eval()
        for p in self.mano_fit.parameters():
            p.requires_grad = False

        self.keypoint_3d_loss = Keypoint3DLoss(loss_type="l1")
        self.mano_parameter_loss = ParameterLoss()

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
        B = b["img"].shape[0]

        feats = self.backbone(b["img"], b["base_img"])
        theta_hat, dtheta, gate = self.head(feats, b["hamer_pose"])
        pred_mano_out = self.mano_fit(b["hamer_shape"], theta_hat)

        pred_kp_3d = mano_to_openpose(pred_mano_out.joints, pred_mano_out.vertices)

        out = {
            "pose": theta_hat,
            "shape": b["hamer_shape"],
            "vertices": pred_mano_out.vertices,
            "kp_3d": pred_kp_3d,
            "dtheta": dtheta,
            "gate": gate,
        }

        return out

    def compute_loss(self, gt_kp_3d, pred_kp_3d, gt_pose, pred_pose, loss_sid=None):
        batch_size = gt_kp_3d.shape[0]
        loss_kp_3d = self.keypoint_3d_loss(pred_kp_3d, gt_kp_3d, pelvis_id=0)

        R_gt = axis_angle_to_matrix(gt_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)
        R_pred = axis_angle_to_matrix(pred_pose.reshape(-1, 3)).view(
            batch_size, -1, 3, 3
        )

        pose_param_loss = self.mano_parameter_loss(R_pred.reshape(-1), R_gt.reshape(-1))

        loss = (self.hparams.loss_weights["keypoints_3d"] * loss_kp_3d) + (
            self.hparams.loss_weights["pose"] * pose_param_loss
        )

        return loss

    def model_step(self, b):
        out = self.forward(b)

        gt_pose = b["gt_pose"]
        gt_mano_out = self.mano_fit(b["hamer_shape"], b["gt_pose"])
        gt_kp_3d = mano_to_openpose(gt_mano_out.joints, gt_mano_out.vertices)

        loss = self.compute_loss(gt_kp_3d, out["kp_3d"], gt_pose, out["pose"])

        target = {
            "pose": b["gt_pose"],
            "shape": b["hamer_shape"],
            "vertices": gt_mano_out.vertices,
            "kp_3d": gt_kp_3d,
        }
        return loss, out, target

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

    def validation_step(self, b, b_idx):
        B = b["hamer_shape"].shape[0]

        start_time = time.time()
        loss, preds, targets = self.model_step(b)
        elapsed_time = int((time.time() - start_time) * 1000)

        mpjpe = calc_mpjpe(preds["kp_3d"], targets["kp_3d"])

        prior_mano_out = self.mano_fit(b["hamer_shape"], b["hamer_pose"])
        prior_kp_3d = mano_to_openpose(prior_mano_out.joints, prior_mano_out.vertices)
        mpjpe_prior = calc_mpjpe(prior_kp_3d, targets["kp_3d"])

        if b_idx % 25 == 0:
            panels = log_examples(
                b["img"],
                b["base_img"],
                targets["vertices"],
                preds["vertices"],
                prior_mano_out.vertices,
                self.mano_fit.faces,
            )
            wandb_panels = []
            for i, panel in enumerate(panels):
                caption = f"Sample{i}: prediction, GT, prior"
                wandb_panels.append(wandb.Image(panel, caption=caption))

            self.logger.experiment.log(
                {
                    f"val/prediction_examples_batch_{b_idx}": wandb_panels,
                    "epoch": int(self.current_epoch),
                },
                step=self.global_step,
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
            "val/mpjpe",
            mpjpe.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/mpjpe-prior",
            mpjpe_prior.mean(),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/val-inference-time",
            elapsed_time / B,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_start(self):
        self._test_outputs = {
            "frame_id": [],
            "pose": [],
            "shape": [],
            "pred_kp_3d": [],
            "gt_kp_3d": [],
            "hamer_kp_3d": [],
        }

    def test_step(self, b, b_idx):
        B = b["hamer_shape"].shape[0]

        start_time = time.time()
        loss, preds, targets = self.model_step(b)
        elapsed_time = int((time.time() - start_time) * 1000)

        mpjpe = calc_mpjpe(preds["kp_3d"], targets["kp_3d"])

        prior_mano_out = self.mano_fit(b["hamer_shape"], b["hamer_pose"])
        prior_kp_3d = mano_to_openpose(prior_mano_out.joints, prior_mano_out.vertices)
        mpjpe_prior = calc_mpjpe(prior_kp_3d, targets["kp_3d"])

        if b_idx % 25 == 0:
            panels = log_examples(
                b["img"],
                b["base_img"],
                targets["vertices"],
                preds["vertices"],
                prior_mano_out.vertices,
                self.mano_fit.faces,
            )
            wandb_panels = []
            for i, panel in enumerate(panels):
                caption = f"Sample{i}: prediction, GT, prior"
                wandb_panels.append(wandb.Image(panel, caption=caption))

            self.logger.experiment.log(
                {
                    f"test/prediction_examples_batch_{b_idx}": wandb_panels,
                    "epoch": int(self.current_epoch),
                },
                step=self.global_step,
            )

        self._test_outputs["frame_id"].append(b["frame_id"])
        self._test_outputs["pose"].append(preds["pose"])
        self._test_outputs["shape"].append(b["hamer_shape"])
        self._test_outputs["pred_kp_3d"].append(preds["kp_3d"])
        self._test_outputs["gt_kp_3d"].append(targets["kp_3d"])
        self._test_outputs["hamer_kp_3d"].append(prior_kp_3d)

        self.log(
            "test/loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/mpjpe",
            mpjpe.mean(),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/mpjpe-prior",
            mpjpe_prior.mean(),
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/val-inference-time",
            elapsed_time / B,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(
            output_dir, f"test_outputs_epoch{self.current_epoch}.npy"
        )

        final_outputs = {}
        for key, val in self._test_outputs.items():
            local = torch.concat(val)
            gathered = self.all_gather(local)
            final_outputs[key] = gathered.detach().cpu().to(torch.float32).numpy()

        np.save(file_path, final_outputs, allow_pickle=True)

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
