"""Geometric utility functions for coordinate transformations and image processing."""

import numpy as np
import cv2
import torch


def matrix_to_axis_angle_np(R, eps: float = 1e-7):
    tr = np.einsum("...ii", R)
    cos = np.clip((tr - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    theta = np.arccos(cos)[..., None]

    s = 0.5 * np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )

    sin_theta = np.linalg.norm(s, axis=-1, keepdims=True)
    factor = np.where(sin_theta > eps, theta / sin_theta, 1.0 + (theta**2) / 6.0)
    aa = factor * s
    return aa


def axis_angle_to_matrix_np(aa: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if aa.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3, got {aa.shape}")

    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2]
    zeros = np.zeros_like(ax)

    K = np.stack([zeros, -az, ay, az, zeros, -ax, -ay, ax, zeros], axis=-1).reshape(
        aa.shape[:-1] + (3, 3)
    )

    theta2 = theta * theta
    theta4 = theta2 * theta2
    s = np.where(theta > eps, np.sin(theta) / theta, 1 - theta2 / 6 + theta4 / 120)
    c2 = np.where(
        theta > eps, (1 - np.cos(theta)) / theta2, 0.5 - theta2 / 24 + theta4 / 720
    )

    s = s[..., None, None]
    c2 = c2[..., None, None]

    I = np.eye(3, dtype=aa.dtype)
    I = np.broadcast_to(I, K.shape)
    K2 = K @ K

    R = I + s * K + c2 * K2
    return R


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-9)
    k = aa / theta
    kx, ky, kz = k.unbind(dim=-1)
    zeros = torch.zeros_like(kx)

    K = torch.stack([zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], dim=-1).reshape(
        *aa.shape[:-1], 3, 3
    )

    sin = torch.sin(theta)[..., None]
    cos = torch.cos(theta)[..., None]

    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    K2 = K @ K
    R = I + sin * K + (1.0 - cos) * K2

    return R


def align_images(frame, curr_points, ref_points):
    H, inliers = cv2.findHomography(
        curr_points, ref_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )

    h, w = frame.shape[:2]
    aligned = cv2.warpPerspective(
        frame,
        H,
        dsize=(w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    transformed_pts = cv2.perspectiveTransform(
        curr_points.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    return aligned, transformed_pts


def project_2d(img, points, cam_t, scaled_focal_length):
    H, W = img.shape[:2]
    fx = fy = float(scaled_focal_length)  # already scaled for this image
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    rvec = np.zeros((3, 1), np.float32)
    tvec = np.asarray(cam_t, np.float32).reshape(3, 1)  # camera translation
    pts_proj, _ = cv2.projectPoints(np.asarray(points, np.float32), rvec, tvec, K, None)
    pts_proj = pts_proj.reshape(-1, 2)

    return pts_proj


def crop_by_points(img, pts, pad=50):
    cols = pts[:, 0]
    rows = pts[:, 1]

    c_min = int(np.floor(cols.min()) - pad)
    c_max = int(np.ceil(cols.max()) + 2 * pad + 1)  # +1 to include the last pixel
    r_min = int(np.floor(rows.min()) - pad)
    r_max = int(np.ceil(rows.max()) + pad + 1)

    h, w = img.shape[:2]
    c_min, c_max = np.clip([c_min, c_max], 0, w)
    r_min, r_max = np.clip([r_min, r_max], 0, h)
    out = img[r_min:r_max, c_min:c_max]

    return out


def mask_by_points(img, pts, dilation_iter=10, dilation_kernel_size=5, color=(0, 0, 0)):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size)
    )
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iter)

    out = img.copy()
    out[dilated == 0] = color

    return out


def procrustes_np(X, Y, mask=None, eps=1e-8):
    N, J, _ = X.shape
    if mask is None:
        mask = np.ones((N, J), dtype=bool)

    w = mask[..., None].astype(X.dtype)
    # weighted means
    wsum = np.clip(w.sum(axis=1, keepdims=True), 1.0, None)
    muX = (X * w).sum(axis=1, keepdims=True) / wsum
    muY = (Y * w).sum(axis=1, keepdims=True) / wsum

    # center and apply mask
    Xc = (X - muX) * w
    Yc = (Y - muY) * w

    # covariance per sample
    C = np.einsum("nij,nik->njk", Xc, Yc)

    # SVD and proper rotation
    U, S, Vt = np.linalg.svd(C)
    R = Vt.transpose(0, 2, 1) @ U.transpose(0, 2, 1)
    detR = np.linalg.det(R) < 0
    if np.any(detR):
        Vt[detR, -1, :] *= -1
        R[detR] = Vt[detR].transpose(0, 2, 1) @ U[detR].transpose(0, 2, 1)

    # isotropic scale
    varX = np.sum(Xc**2, axis=(1, 2)) + eps
    s = (S.sum(axis=1)) / varX

    # translation
    muX_rot = np.einsum("nij,nkj->nki", R, muX).squeeze(1)
    t = muY.squeeze(1) - s[:, None] * muX_rot

    # apply to all joints
    X_rot = np.einsum("nij,nkj->nki", R, X)
    X_aligned = s[:, None, None] * X_rot + t[:, None, :]

    return X_aligned, s, R, t
