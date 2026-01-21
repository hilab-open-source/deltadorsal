"""Loss functions for pose and parameter estimation."""

import torch
import torch.nn as nn


class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(
        self,
        pred_keypoints_3d: torch.Tensor,
        gt_keypoints_3d: torch.Tensor,
        pelvis_id: int = 0,
    ):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 3] containing the predicted 3D keypoints (B: batch_size, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 3] containing the ground truth 3D keypoints
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[
            :, pelvis_id, :
        ].unsqueeze(1)
        gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, pelvis_id, :].unsqueeze(
            1
        )
        loss = (self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):
    def __init__(self):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute MANO parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        loss_param = self.loss_fn(pred_param, gt_param)
        return loss_param.sum()
