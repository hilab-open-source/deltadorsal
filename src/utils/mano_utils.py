"""MANO-related utilities for hand model processing."""

import numpy as np
import cv2
import torch
import torch.nn as nn

from utils.geom_utils import procrustes_np


def rasterize_visibility(tris_pix, tris_depth, front_facing, H, W):
	"""
	barycentric visibility testing

	tris_pix:     (N,3,2) float — pixel coords of each triangle's verts
	tris_depth:   (N,3)   float — their depths (Z) in camera space
	front_facing: (N,)    bool  — which faces to *mark* visible
	H, W:         ints    — image height & width

	Returns:
	  visible:    (N,) bool — True if any fragment of that face passed the z-test
	"""
	N = tris_pix.shape[0]
	zbuf    = np.full((H, W), np.inf, dtype=np.float32)
	visible = np.zeros(N,     dtype=bool)

	for i in range(N):
		pix    = tris_pix[i]
		depths = tris_depth[i]

		# screen‐space bbox
		xs, ys = pix[:,0], pix[:,1]
		minx = max(int(np.floor(xs.min())), 0)
		maxx = min(int(np.ceil (xs.max())), W-1)
		miny = max(int(np.floor(ys.min())), 0)
		maxy = min(int(np.ceil (ys.max())), H-1)
		if minx > maxx or miny > maxy:
			continue

		# pixel‐center grid in bbox
		Xs = np.arange(minx, maxx+1)
		Ys = np.arange(miny, maxy+1)
		PX, PY = np.meshgrid(Xs, Ys)
		PXc = PX.ravel() + 0.5
		PYc = PY.ravel() + 0.5


		# precompute barycentric denominator (twice signed area)
		(x0,y0), (x1, y1), (x2, y2) = pix
		den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
		if den == 0:
			continue

		alpha = ((y1 - y2)*(PXc - x2) + (x2 - x1)*(PYc - y2)) / den
		beta  = ((y2 - y0)*(PXc - x2) + (x0 - x2)*(PYc - y2)) / den
		gamma = 1.0 - alpha - beta

		# mask of points inside triangle
		mask = (alpha >= 0) & (beta >= 0) & (gamma >= 0)
		if not mask.any():
			continue

		# perspective corrective interpolation
		Z_inv = alpha[mask]*(1/depths[0]) + beta[mask]*(1/depths[1]) + gamma[mask]*(1/depths[2])
		Z = 1.0 / Z_inv

		pxs = PX.ravel()[mask].astype(int)
		pys = PY.ravel()[mask].astype(int)

		old = zbuf[pys, pxs]
		keep = Z < old

		# update z‐buffer
		if keep.any():
			zbuf[pys[keep], pxs[keep]] = Z[keep]
			if front_facing[i]:
				visible[i] = True

	return visible

def compute_occluded_faces(vertices:np.ndarray, faces:np.ndarray, K:np.ndarray, d:np.ndarray, R:np.ndarray, t: np.ndarray, H:int, W:int) -> tuple[np.ndarray, np.ndarray]:
	# for single mano at a time
	Rvec, _ = cv2.Rodrigues(R)

	img_pts, jacobian = cv2.projectPoints(vertices, Rvec, t, K, d)
	V_pix = img_pts.reshape(-1, 2)

	V_cam= (vertices @ R.T) + t.reshape(3,)
	# V_world = (R.T @ (V_cam - t).T).T
	depths = V_cam[:,2]

	# tris_world = V_world[faces]
	tris_cam = V_cam[faces]
	tris_pix = V_pix[faces]
	tris_depth = depths[faces]

	v0 = tris_cam[:,1] - tris_cam[:,0]
	v1 = tris_cam[:,2] - tris_cam[:,0]
	normals = np.cross(v0, v1)
	n = normals / np.linalg.norm(normals, axis=1, keepdims=True)
	view_dir = np.array([0,0,1], dtype=np.float32)

	front_facing = (np.dot(n, view_dir) < (3 * np.pi / 180)).reshape(-1)  # boolean mask, True = keep

	visible = rasterize_visibility(tris_pix, tris_depth, front_facing, H, W)

	return visible, V_cam

def perc_occluded_per_label(visible, faces, vertices, labels):
	# visibility mask + faces into % occluded per label
	# label is labeling of faces taken as input

	# area of triangle per mano face
	tris = vertices[:, faces, :]
	v0 = tris[:, :, 0, :] - tris[:, :, 1, :]
	v1 = tris[:, :, 0, :] - tris[:, :, 2, :]
	tri_areas = .5 * np.linalg.norm(np.cross(v0, v1, axis=-1), axis=-1)

	labels = np.array(labels)
	unique_labels = np.unique(labels)

	out = {}
	for u in unique_labels:
		face_label_mask = (labels == u)
		mask = visible & face_label_mask[np.newaxis,:]
		num = (tri_areas * mask).sum(axis=-1)
		den = (tri_areas * face_label_mask[np.newaxis,:]).sum(axis=-1)
		out[u] = num / den

	return out

def mano_to_openpose(mano_joints: torch.Tensor, mano_vertices: torch.Tensor) -> torch.Tensor:
	B = mano_joints.shape[0]
	device = mano_joints.device
	dtype = mano_joints.dtype

	out = torch.zeros(B, 21, 3, device=device, dtype=dtype)

	# map the 16 MANO joints into the first 16 OpenPose slots
	# src indices in MANO, dst indices in OpenPose:
	#                        wrist   thumb       index      middle        ring         pinky
	joint_src = torch.tensor([ 0,  13, 14, 15,  1, 2, 3,   4, 5,  6,   10, 11, 12,   7,  8,  9],
					   device=device)
	joint_dst = torch.tensor([ 0,  1,  2,  3,   5, 6, 7,   9, 10, 11,  13, 14, 15,   17, 18, 19],
					   device=device)

	tip_src = torch.tensor([743, 333, 443, 554, 671], device=device)
	tip_dst = torch.tensor([4  , 8  , 12 , 16 , 20 ], device=device)

	out[:, joint_dst, :] = mano_joints[:, joint_src, :]
	out[:, tip_dst, :] = mano_vertices[:, tip_src, :]

	return out

def openpose_to_mano(openpose_kps, mano_model, lr=1e-2, n_iters=2000, reg_weight=1e-4, device=None, verbose=False):
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	B = openpose_kps.shape[0]

	mse_loss = nn.MSELoss()

	global_orient = torch.zeros((B, 3), requires_grad=True, device=device)
	hand_pose     = torch.zeros((B, 45), requires_grad=True, device=device)
	trans         = torch.zeros((B, 3), requires_grad=True, device=device)
	betas         = torch.zeros((B, 10), requires_grad=True, device=device)

	optimizer = torch.optim.Adam([global_orient, hand_pose, trans,betas], lr=lr)
	for step in range(n_iters):
		optimizer.zero_grad()

		output = mano_model(
				betas=betas,
				global_orient=global_orient,
				hand_pose=hand_pose,
				transl=trans
			)

		pred_openpose = mano_to_openpose(output.joints, output.vertices)
		data_loss = mse_loss(pred_openpose, openpose_kps)

		shape_reg = reg_weight * betas.pow(2).mean()
		pose_reg = .25 * reg_weight * hand_pose.pow(2).mean()
		loss = data_loss + pose_reg + shape_reg

		loss.backward()
		optimizer.step()
		if verbose:
			if step % 250 == 0 or step == n_iters-1:
				print(f"step {step}/{n_iters}, loss={loss.item():.6f}")

	with torch.no_grad():
		final_out = mano_model(
			betas=betas,
			global_orient=global_orient,
			hand_pose=hand_pose,
			transl=trans
		)

	return hand_pose.detach(), betas.detach(), trans.detach(), global_orient.detach(), final_out


def calc_mpjpe(pred_joints: torch.tensor, gt_joints: torch.tensor) -> torch.tensor:
	mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1)
	return mpjpe

def calc_auc_joints_np(preds, gts, tau_max=50.0, num_thresholds=501):
	errs = np.linalg.norm(preds - gts, axis=-1)
	errs = errs[~np.isnan(errs)].ravel()
	taus = np.linspace(0.0, float(tau_max), int(num_thresholds))
	pck = (errs[None, :] < taus[:, None]).mean(axis=1)
	auc = np.trapezoid(pck, taus) / float(tau_max)
	return float(auc)

_ANGLE_TRIPLETS = np.array([
	# thumb
	[0,  1,  2],
	[1,  2,  3],
	[2,  3,  4],
	# index
	[0,  5,  6],
	[5,  6,  7],
	[6,  7,  8],
	# middle
	[0,  9, 10],
	[9, 10, 11],
	[10, 11, 12],
	# ring
	[0, 13, 14],
	[13, 14, 15],
	[14, 15, 16],
	# pinky
	[0, 17, 18],
	[17, 18, 19],
	[18, 19, 20],
], dtype=np.int64)

def _angles_from_points(points):
	# Gather (parent, joint, child) coords
	p = points[..., _ANGLE_TRIPLETS[:, 0], :]
	j = points[..., _ANGLE_TRIPLETS[:, 1], :]
	c = points[..., _ANGLE_TRIPLETS[:, 2], :]

	# bone vectors pointing away from the joint
	v1 = p - j
	v2 = c - j

	# normalize with safe eps to avoid divide-by-zero
	eps = 1e-8
	v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True)
	v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True)
	v1u = v1 / np.maximum(v1_norm, eps)
	v2u = v2 / np.maximum(v2_norm, eps)

	# angle via arccos of clipped dot
	dots = np.sum(v1u * v2u, axis=-1)
	dots = np.clip(dots, -1.0, 1.0)
	ang = np.arccos(dots)
	return ang # radians

def calc_mpjae(pred, gt,  degrees=True, ):
	pred = np.asarray(pred, dtype=np.float64)
	gt   = np.asarray(gt,   dtype=np.float64)
	assert pred.shape == gt.shape and pred.ndim == 3 and pred.shape[1:] == (21, 3), \
		"pred and gt must be (N, 21, 3)"

	# Compute angles
	ang_pred = _angles_from_points(pred)
	ang_gt   = _angles_from_points(gt)

	# Absolute error
	err = np.abs(ang_pred - ang_gt)

	if degrees:
		err = np.degrees(err)
	return err.mean(axis=1), err.mean(axis=0), err

def calc_pa_mpjpe(pred, gt):
	pred_aligned, _, _, _ = procrustes_np(pred, gt)
	err = np.linalg.norm(pred_aligned - gt, axis=-1)

	return err.mean(axis=1), err.mean(axis=0), err
