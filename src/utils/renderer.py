"""Mesh rendering utilities for visualization."""

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
import numpy as np
import pyrender
import trimesh


class Renderer:
    def __init__(
        self, faces: np.array, mesh_color=(0.77, 0.84, 0.95), bg_color=(1.0, 1.0, 1.0)
    ):
        faces_new = np.array(
            [
                [92, 38, 234],
                [234, 38, 239],
                [38, 122, 239],
                [239, 122, 279],
                [122, 118, 279],
                [279, 118, 215],
                [118, 117, 215],
                [215, 117, 214],
                [117, 119, 214],
                [214, 119, 121],
                [119, 120, 121],
                [121, 120, 78],
                [120, 108, 78],
                [78, 108, 79],
            ]
        )
        faces = np.concatenate([faces, faces_new], axis=0)

        self.faces = faces
        self.mesh_color = mesh_color
        self.bg_color = bg_color

    @staticmethod
    def _make_trimesh(verts, faces, color):
        if torch.is_tensor(verts):
            verts = verts.detach().cpu().numpy()
        vcolor = np.tile(np.array([*color, 1.0], dtype=np.float32), (verts.shape[0], 1))
        return trimesh.Trimesh(
            vertices=verts.copy(),
            faces=faces.copy(),
            vertex_colors=vcolor,
            process=False,
        )

    @staticmethod
    def _make_trimesh(verts, faces, color):
        if torch.is_tensor(verts):
            verts = verts.detach().cpu().numpy()
        vcolor = np.tile(np.array([*color, 1.0], dtype=np.float32), (verts.shape[0], 1))
        return trimesh.Trimesh(
            vertices=verts.copy(),
            faces=faces.copy(),
            vertex_colors=vcolor,
            process=False,
        )

    @staticmethod
    def _center_radius(verts):
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        center = 0.5 * (vmin + vmax)
        radius = 0.5 * np.max(vmax - vmin) + 1e-9
        return center, float(radius)

    @staticmethod
    def _look_at(eye, target, up=(0, 1, 0)):
        eye = np.asarray(eye, np.float32)
        target = np.asarray(target, np.float32)
        up = np.asarray(up, np.float32)

        z = target - eye
        z /= np.linalg.norm(z) + 1e-9

        x = np.cross(z, up)
        x /= np.linalg.norm(x) + 1e-9

        y = np.cross(x, z)

        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = -z  # camera looks down -Z
        T[:3, 3] = eye
        return T

    @staticmethod
    def _add_rim_lights(scene, intensity=3.0):
        dl1 = pyrender.DirectionalLight(color=np.ones(3), intensity=float(intensity))
        dl2 = pyrender.DirectionalLight(color=np.ones(3), intensity=float(intensity))
        dl3 = pyrender.DirectionalLight(color=np.ones(3), intensity=float(intensity))

        p1 = trimesh.transformations.rotation_matrix(
            np.deg2rad(45), [1, 0, 0]
        ) @ trimesh.transformations.rotation_matrix(np.deg2rad(35), [0, 1, 0])
        p2 = trimesh.transformations.rotation_matrix(
            np.deg2rad(-30), [1, 0, 0]
        ) @ trimesh.transformations.rotation_matrix(np.deg2rad(-45), [0, 1, 0])
        p3 = trimesh.transformations.rotation_matrix(
            np.deg2rad(0), [1, 0, 0]
        ) @ trimesh.transformations.rotation_matrix(np.deg2rad(120), [0, 1, 0])

        scene.add_node(pyrender.Node(light=dl1, matrix=p1))
        scene.add_node(pyrender.Node(light=dl2, matrix=p2))
        scene.add_node(pyrender.Node(light=dl3, matrix=p3))

    @staticmethod
    def _make_shadow_disc(radius, alpha=0.18, scale=1.35, y=-0.0):
        r = float(radius) * float(scale)
        disc = trimesh.creation.cylinder(radius=r, height=1e-3, sections=64)
        disc.apply_translation([0, y, 0])
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0, 0, 0, float(alpha)),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
        )
        return pyrender.Mesh.from_trimesh(disc, material=mat)

    @staticmethod
    def _rgba_to_rgb(rgba, bg_color=(1.0, 1.0, 1.0)):
        if rgba.shape[-1] != 4:
            return rgba  # already RGB
        a = rgba[..., 3:4]
        bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
        return rgba[..., :3] * a + bg * (1.0 - a)

    @staticmethod
    def _prepare_output(img_rgb_float, out_format="rgb_uint8"):
        """
        img_rgb_float: (H,W,3) float32 in [0,1] (RGB)
        out_format: "rgb_uint8" | "bgr_uint8" | "rgb_float" | "bgr_float"
        """
        img = img_rgb_float
        if out_format.startswith("bgr"):
            img = img[..., ::-1]
        if out_format.endswith("uint8"):
            img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return img

    def render_view(
        self,
        verts,  # (V,3) world-space
        render_res=(512, 512),
        yfov_deg=30.0,
        fit_margin=1.45,
        side_view=True,
        top_yaw_deg=-70.0,
        side_elev_deg=-60.0,
        add_shadow=True,
        shadow_alpha=0.18,
        shadow_scale=1.35,
        light_intensity=3.0,
        bg_color=None,
        mesh_color=None,
    ):
        mesh_color = self.mesh_color if mesh_color is None else mesh_color
        bg = self.bg_color if bg_color is None else bg_color

        if torch.is_tensor(verts):
            verts = verts.detach().cpu().numpy()
        tri = self._make_trimesh(verts, self.faces, mesh_color)
        center, radius = self._center_radius(verts)
        tri.apply_translation(-center)  # recenter

        scene = pyrender.Scene(bg_color=[*bg, 0.0], ambient_light=(0.25,) * 3)
        pmesh = pyrender.Mesh.from_trimesh(tri, smooth=True)

        scene.add(pmesh)

        self._add_rim_lights(scene, intensity=light_intensity)
        if add_shadow:
            y_shadow = -0.55 * radius
            scene.add(
                self._make_shadow_disc(radius, shadow_alpha, shadow_scale, y_shadow)
            )

        yfov = np.deg2rad(yfov_deg)
        cam = pyrender.PerspectiveCamera(yfov=yfov)
        dist = fit_margin * radius / np.tan(yfov / 2.0)

        if not side_view:
            # top
            yaw = np.deg2rad(top_yaw_deg)
            eye_side = np.array(
                [np.sin(yaw) * dist, 0.05 * radius, np.cos(yaw) * dist], np.float32
            )
            scene.add(cam, pose=self._look_at(eye_side, target=np.zeros(3, np.float32)))
        else:
            # side
            elev = np.deg2rad(side_elev_deg)
            azim = np.deg2rad(-45.0)
            eye_top = np.array(
                [
                    np.cos(elev) * np.sin(azim) * dist,
                    np.sin(elev) * dist,
                    np.cos(elev) * np.cos(azim) * dist,
                ],
                np.float32,
            )
            scene.add(cam, pose=self._look_at(eye_top, target=np.zeros(3, np.float32)))

        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0], viewport_height=render_res[1]
        )
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        return color

    def render_overlay(
        self,
        image_rgb,
        verts,
        K,
        R=None,
        t=None,
        bg_transparent=True,
        light_intensity=2.5,
        mesh_color=None,
    ):
        """
        Returns the composited image (H,W,3) float32 in [0,1] by default.
        If return_rgba=True, returns (rgba_render, composite).
        """
        mesh_color = self.mesh_color if mesh_color is None else mesh_color

        # image prep
        img = np.asarray(image_rgb)
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        H, W = img.shape[:2]

        # transform verts to CAMERA coords if R,t provided
        if torch.is_tensor(verts):
            verts = verts.detach().cpu().numpy()
        if R is not None and t is not None:
            R = np.asarray(R, np.float32).reshape(3, 3)
            t = np.asarray(t, np.float32).reshape(
                3,
            )
            verts_cam = (R @ verts.T).T + t[None, :]
        else:
            verts_cam = verts.copy()

        # OpenCV -> OpenGL (pyrender) coords: flip Y,Z
        M_cv2gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        verts_cam = (M_cv2gl @ verts_cam.T).T

        # build mesh in CAMERA space; camera at origin
        tri = self._make_trimesh(verts_cam, self.faces, mesh_color)
        pmesh = pyrender.Mesh.from_trimesh(tri, smooth=True)

        # scene
        bg = [*self.bg_color, 0.0] if bg_transparent else [*self.bg_color, 1.0]
        scene = pyrender.Scene(bg_color=bg, ambient_light=(0.25,) * 3)
        scene.add(pmesh)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cam = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy, znear=1e-3, zfar=1e3
        )
        scene.add(
            cam, pose=np.eye(4, dtype=np.float32)
        )  # camera at origin in CAMERA frame

        # camera-anchored lights
        self._add_rim_lights(scene, intensity=light_intensity)

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        rgba = rgba.astype(np.float32) / 255.0

        # default: alpha-composite over the image
        out = rgba[..., :3] * rgba[..., 3:4] + img * (1.0 - rgba[..., 3:4])

        return out * 255
