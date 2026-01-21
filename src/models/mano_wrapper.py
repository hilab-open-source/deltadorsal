"""MANO model wrapper for pose-only estimation."""

import torch
import smplx
from smplx.utils import MANOOutput


class MANOPoseOnly(smplx.MANO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, betas, pose, **kwargs) -> MANOOutput:
        B = betas.shape[0]
        mano_output = super().forward(
            betas=betas,
            hand_pose=pose,
            transl=torch.zeros(B, 3, device=betas.device, dtype=betas.dtype),
            global_orient=torch.zeros(B, 3, device=betas.device, dtype=betas.dtype),
            **kwargs,
        )
        return mano_output
