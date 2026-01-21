"""Visualization utilities for hand pose and force estimation results."""

from PIL import Image
import cv2

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_HAND_EDGES = [
	(0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
	(0, 5), (5, 6), (6, 7), (7, 8),      # Index
	(0, 9), (9, 10), (10, 11), (11, 12), # Middle
	(0, 13), (13, 14), (14, 15), (15, 16), # Ring
	(0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

HAND_JOINT_NAMES = [
	"wrist", # 0
	"thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", # 1–4
	"index_mcp", "index_pip", "index_dip", "index_tip", # 5–8
	"middle_mcp", "middle_pip", "middle_dip", "middle_tip", # 9–12
	"ring_mcp", "ring_pip", "ring_dip", "ring_tip", # 13–16
	"pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip" # 17–20
]

def plot_openpose_hand_3d(kps, elev=30, azim=45, point_size=30, line_width=2):
	"""
	Plot a 3D OpenPose-style hand skeleton.

	Args:
		kps (array-like, shape (21,3)): x,y,z coords of the 21 keypoints.
		elev (float): elevation angle in the z plane.
		azim (float): azimuth angle in the x,y plane.
		point_size (int): scatter point size.
		line_width (int): width of the bone lines.
	"""
	kps = np.asarray(kps)
	fig = plt.figure(figsize=(6,6))
	ax = fig.add_subplot(111, projection='3d')
	# draw bones
	for i, j in _HAND_EDGES:
		xs, ys, zs = [kps[i,0], kps[j,0]], [kps[i,1], kps[j,1]], [kps[i,2], kps[j,2]]
		ax.plot(xs, ys, zs, 'k-', lw=line_width)

	text_size=8
	text_offset=0.002
	for idx, (x, y, z) in enumerate(kps):
		ax.text(x, y, z + text_offset, str(idx),
			size=text_size, zorder=1, color='black')
	# draw joints
	ax.scatter(kps[:,0], kps[:,1], kps[:,2], c='r', s=point_size)
	# set view
	ax.view_init(elev=elev, azim=azim)
	# equal axes
	ax.auto_scale_xyz(kps[:,0], kps[:,1], kps[:,2])
	ax.set_xlim(0, .4)
	ax.set_ylim(0, .4)
	ax.set_zlim(0, .4)
	# ax.set_axis_off()
	plt.tight_layout()
	plt.show()
	return ax

def vis_mano(V_cam, faces, face_colors=None, labels=False, return_img=False):
	if face_colors is None:
		face_colors = ["lightgray"] * len(faces)

	fig = plt.figure(figsize=(8, 8))
	ax  = fig.add_subplot(111, projection='3d')

	mesh = Poly3DCollection(
		V_cam[faces],
		facecolors=face_colors,
		edgecolors='k',
		linewidths=0.2,
		alpha=0.9
	)
	ax.add_collection3d(mesh)

	if labels:
		centroids = V_cam[faces].mean(axis=1)
		for i, (x,y,z) in enumerate(centroids):
			ax.text(x, y, z, str(i), color='black', fontsize=8, ha='center', va='center')
	# Auto‐scale to the extents of V_cam
	xyz = V_cam.reshape(-1, 3)
	ax.auto_scale_xyz(xyz[:,0], xyz[:,1], xyz[:,2])
	ax.set_box_aspect([1,1,1])


	ax.view_init(elev=-90, azim=90)
	plt.tight_layout()

	if return_img:
		plt.close(fig)
		ax.axis("off")
		ax.set_position([0,0,1,1])
		ax.set_axis_off()
		ax.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])
		for spine in ax.spines.values():
			spine.set_visible(False)
		return fig
	else:
		ax.set_xlabel('X (cam)')
		ax.set_ylabel('Y (cam)')
		ax.set_zlabel('Z (cam)')
		ax.set_title('MANO Mesh — Front-Facing Faces in Red')
		plt.show()


def unnormalize_img(images):
	images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
	images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

	return images


def log_examples(imgs, bases, target_vertices, pred_vertices, prior_vertices, faces, n_per_batch=3, W=256, H=256):
	curr, base = imgs, bases

	batch_size = curr.shape[0]

	if n_per_batch > batch_size:
		n_per_batch = batch_size


	curr = unnormalize_img(curr)
	base = unnormalize_img(base)

	panels = []
	for i in range(0, batch_size, batch_size // n_per_batch):
		panel = Image.new("RGB", (W * 3, H*2))
		panel.paste(TF.to_pil_image(base[i].cpu()).resize((W,H)), (0,0))
		panel.paste(TF.to_pil_image(curr[i].cpu()).resize((W,H)), (W,0))

		for v_idx, v in enumerate((pred_vertices, target_vertices, prior_vertices)):
			v = v[i].cpu().numpy()
			fig = vis_mano(v, faces, return_img=True)
			fig.canvas.draw()
			w, h = fig.canvas.get_width_height()
			buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
			vis = Image.fromarray(buf.reshape(h, w, 4)).resize((W, H))
			panel.paste(vis, (v_idx * W, H))
			panels.append(panel)

	return panels


def visualize_mano_on_image(
	image: np.ndarray,
	verts: np.ndarray,
	K: np.ndarray,
	R_w2c: np.ndarray,
	t_w2c: np.ndarray,
	d: np.ndarray,
	faces,
	ann_pts: np.ndarray = None,
	ann_labels: list[str] = None,
	mesh_color=(0, 255, 0),
	point_color=(0, 0, 255),
	line_thickness=1,
	point_radius=5
):
	'''
	Projects MANO mesh vertices and optional annotation points onto an image
	and visualizes them.

	Args:
		image (np.ndarray): The 2D image (OpenCV BGR format).
		verts (np.ndarray): Nx3 array of 3D mesh vertices (in world coordinates).
		K (np.ndarray): 3x3 camera intrinsic matrix.
		R_w2c (np.ndarray): 3x3 rotation matrix from world to camera.
		t_w2c (np.ndarray): 3x1 translation vector from world to camera.
		d (np.ndarray): Camera distortion coefficients (e.g., (1,4) or (1,5)).
						Use np.zeros((4,1)) if no distortion.
		save_path (str): Path to save the output image.
		ann_pts (np.ndarray, optional): Mx3 array of 3D annotation points (in world coordinates).
										Defaults to None.
		mesh_color (tuple): BGR color for the mesh wireframe.
		point_color (tuple): BGR color for the annotation points.
		line_thickness (int): Thickness for mesh lines.
		point_radius (int): Radius for annotation points.
	'''
	verts = verts.astype(np.float32)
	K = K.astype(np.float32)
	R_w2c = R_w2c.astype(np.float32)
	t_w2c = t_w2c.astype(np.float32)
	d = d.astype(np.float32)

	rvec_w2c, _ = cv2.Rodrigues(R_w2c)

	# project to 2d
	reprojected_verts_2d, _ = cv2.projectPoints(verts, rvec_w2c, t_w2c, K, d)
	reprojected_verts_2d = reprojected_verts_2d.reshape(-1, 2).astype(int)

	output_image = image.copy()

	for face in faces:
		p1 = reprojected_verts_2d[face[0]]
		p2 = reprojected_verts_2d[face[1]]
		p3 = reprojected_verts_2d[face[2]]

		cv2.line(output_image, tuple(p1), tuple(p2), mesh_color, line_thickness, cv2.LINE_AA)
		cv2.line(output_image, tuple(p2), tuple(p3), mesh_color, line_thickness, cv2.LINE_AA)
		cv2.line(output_image, tuple(p3), tuple(p1), mesh_color, line_thickness, cv2.LINE_AA)

	if ann_pts is not None:
		ann_pts = ann_pts.astype(np.float32)
		reprojected_ann_pts_2d, _ = cv2.projectPoints(ann_pts, rvec_w2c, t_w2c, K, d)
		reprojected_ann_pts_2d = reprojected_ann_pts_2d.reshape(-1, 2).astype(int)

		for i, (x, y) in enumerate(reprojected_ann_pts_2d):
			cv2.circle(output_image, (x, y), point_radius, point_color, -1)

			if ann_labels is not None:
				text_org = (x, y - point_radius - 2)
				cv2.putText(
					output_image,
					str(ann_labels[i]),
					text_org,
					fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=2,
					color=point_color,
					thickness=2,
					lineType=cv2.LINE_AA
				)

	return output_image

def plot_multitask_confusion_matrix(y_true, y_pred):
	"""
	Generates and plots a confusion matrix for the multitask finger activation task.

	Args:
		y_true (torch.Tensor): The ground truth finger force values (B, 5).
		act_logits (torch.Tensor): The predicted finger activation logits (B, 5).
		tau (float): The threshold to determine if a finger is "active".
	"""
	y_true_np = y_true.detach().cpu().to(torch.long).numpy().flatten()
	y_pred_np = y_true.detach().cpu().to(torch.long).numpy().flatten()

	cm = confusion_matrix(y_true_np, y_pred_np, labels=[0,1])

	fig = plt.figure(figsize=(6, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
				xticklabels=['Inactive', "Soft", "Medium", "Hard"],
				yticklabels=['Inactive', "Soft", "Medium", "Hard"])
	plt.title('Confusion Matrix for Finger Activation')
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	fig.canvas.draw()
	w, h = fig.canvas.get_width_height()
	buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
	vis = Image.fromarray(buf.reshape(h, w, 4))

	return vis
