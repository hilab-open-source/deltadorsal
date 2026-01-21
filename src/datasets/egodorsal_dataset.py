"""Dataset classes for ego-dorsal hand pose and force estimation."""

from pathlib import Path
import json
from typing import Literal
import random

import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset

from utils.geom_utils import (
    matrix_to_axis_angle_np,
    align_images,
    crop_by_points,
    project_2d,
)

ALIGNED_IDX = [0, 1, 5, 17]
CROP_IDX = [0, 1, 2, 5, 9, 13, 17]


class EgoDorsalBaseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        selected_participants: list[int] = [],
        split: Literal["train", "val", "test", "all"] = "all",
        sample_ratio=1,
        train=True,
        random_base=True,
        align_kps=ALIGNED_IDX,
        crop_kps=CROP_IDX,
        online=False,
    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.train = train
        self.random_base = random_base
        self.split = split
        self.align_kps = align_kps
        self.crop_kps = crop_kps
        self.online = online
        self.sample_ratio = sample_ratio
        self.selected_participants = selected_participants

        # overrides the split
        if self.selected_participants:
            self.split = "all"

        with open(self.data_dir / "trials.json", "r") as f:
            trials_json = json.load(f)
        self.trials_dict = {t["trial_id"]: t for t in trials_json["trials"]}

        for key in self.trials_dict.keys():
            self.trials_dict[key]["K"] = np.array(self.trials_dict[key]["K"])
            self.trials_dict[key]["d"] = np.array(self.trials_dict[key]["d"])
            self.trials_dict[key]["world2cam"] = np.array(
                self.trials_dict[key]["world2cam"]
            )

        with open(self.data_dir / "bases.json", "r") as f:
            bases_json = json.load(f)
        self.bases_dict = {b["p_id"]: b["bases"] for b in bases_json["participants"]}

        if self.split == "all":
            frames_json_name = "frames.json"
        else:
            frames_json_name = f"{split}.json"

        with open(self.data_dir / frames_json_name, "r") as f:
            frames_json = json.load(f)
        self.frames = frames_json["frames"]

        # select one participant
        if self.selected_participants:
            selected_trials = []
            for trial_id, trial in self.trials_dict.items():
                if trial["p_id"] in self.selected_participants:
                    selected_trials.append(trial_id)

            new_frames = [f for f in self.frames if f["trial_id"] in selected_trials]
            self.frames = new_frames

        sample_step = int(len(self.frames) / (len(self.frames) * self.sample_ratio))
        self.frames = self.frames[::sample_step]

        self.frames_dir = self.data_dir / frames_json["frames_dir"]
        self.bases_dir = self.data_dir / bases_json["bases_dir"]

        self.all_participants = self.bases_dict.keys()

    def __len__(self):
        return len(self.frames)

    def online_img_alignment(self, img_path, hamer_ann, base, K, d):
        orig_img = cv2.imread(img_path)

        # getting the bases
        base_hamer_ann = np.load(
            self.bases_dir / base["hamer_path"], allow_pickle=True
        ).item()
        base_img = cv2.imread(str(self.bases_dir / base["img_path"]))

        h, w = orig_img.shape[:2]

        orig_img = cv2.undistort(orig_img, K, d)
        base_img = cv2.undistort(base_img, K, d)
        # alignment by hamer kps
        scaled_focal_length = (
            K[0, 0] / 256 * max(w, h)
        )  # 256 is the size of the internal image resolution in hamer
        hamer_kp_2d = project_2d(
            orig_img,
            hamer_ann["keypoints_3d"][0],
            hamer_ann["cam_t"],
            scaled_focal_length,
        )
        base_hamer_kp_2d = project_2d(
            orig_img,
            base_hamer_ann["keypoints_3d"][0],
            base_hamer_ann["cam_t"],
            scaled_focal_length,
        )

        base_img = align_images(
            base_img, base_hamer_kp_2d[self.align_kps], hamer_kp_2d[self.align_kps]
        )

        orig_img = crop_by_points(orig_img, hamer_kp_2d[self.crop_kps])
        base_img = crop_by_points(base_img, hamer_kp_2d[self.crop_kps])

        return orig_img, base_img

    def get_mano_visualization_example(self, idx):
        frame = self.frames[idx]
        trial = self.trials_dict[frame["trial_id"]]
        ann = np.load(self.frames_dir / frame["ann_path"], allow_pickle=True).item()
        hamer_ann = np.load(
            self.frames_dir / frame["hamer_path"], allow_pickle=True
        ).item()

        img_path = str(self.frames_dir / frame["img_path"])

        K, d = np.array(trial["K"]), np.array(trial["d"])

        Rt = np.array(trial["world2cam"])
        R = Rt[:, :3]
        t = Rt[:, 3]

        return cv2.imread(str(img_path)), img_path, ann, hamer_ann, R, t, K, d

    def load_base_imgs(self, trial, frame, p_id, hamer_ann):
        # preprocesses and loads the base
        if self.online:
            if self.random_base:
                base = random.choice(self.bases_dict[p_id])
            else:
                base = self.bases_dict[p_id][0]
            K, d = trial["K"], trial["d"]
            img_path = str(self.frames_dir / frame["img_path"])
            orig_img, base_img = self.online_img_alignment(
                img_path, hamer_ann, base, K, d
            )

            base_anns = np.load(
                self.bases_dir / base["hamer_path"], allow_pickle=True
            ).item()
            base_hamer_shape = base_anns["betas"][0]
        else:
            orig_img = cv2.imread(self.frames_dir / frame["cropped_img_path"])
            base_img = cv2.imread(self.frames_dir / frame["cropped_base_path"])
            base_anns = np.load(
                self.bases_dir
                / self.bases_dict[p_id][frame["cropped_base_idx"]]["hamer_path"],
                allow_pickle=True,
            ).item()
            base_hamer_shape = base_anns["betas"][0]

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        orig_img_pil = Image.fromarray(orig_img)
        base_img_pil = Image.fromarray(base_img)

        return orig_img_pil, base_img_pil, base_hamer_shape

    def __getitem__(self, idx):
        raise NotImplementedError("Base class get item not implemented")


class EgoDorsalPoseDataset(EgoDorsalBaseDataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        selected_participants: list[int] = [],
        split: Literal["train", "val", "test", "all"] = "all",
        sample_ratio=1,
        train=True,
        random_base=True,
        align_kps=ALIGNED_IDX,
        crop_kps=CROP_IDX,
        online=False,
    ):
        super().__init__(
            data_dir,
            transforms,
            selected_participants,
            split,
            sample_ratio,
            train,
            random_base,
            align_kps,
            crop_kps,
            online,
        )

    def __getitem__(self, idx):
        frame = self.frames[idx]
        trial = self.trials_dict[frame["trial_id"]]
        p_id = trial["p_id"]

        ann = np.load(self.frames_dir / frame["ann_path"], allow_pickle=True).item()
        hamer_ann = np.load(
            self.frames_dir / frame["hamer_path"], allow_pickle=True
        ).item()

        orig_img_pil, base_img_pil, base_hamer_shape = self.load_base_imgs(
            trial, frame, p_id, hamer_ann
        )

        orig_img_pil = self.transforms(orig_img_pil)
        base_img_pil = self.transforms(base_img_pil)

        # TODO: change this depending on the format of the annotations
        hamer_pose = matrix_to_axis_angle_np(hamer_ann["hand_pose"].squeeze()).reshape(
            -1, 3
        )

        ret = {
            "frame_id": frame["frame_id"],
            "p_id": p_id,
            "img": orig_img_pil,
            "base_img": base_img_pil,
            "hamer_pose": hamer_pose.reshape(45),
            "hamer_shape": base_hamer_shape,
            "gt_pose": ann["pose"].reshape(45),
        }

        return ret


class EgoDorsalForceDataset(EgoDorsalBaseDataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        selected_participants: list[int] = [],
        split: Literal["train", "val", "test", "all"] = "all",
        sample_ratio=1,
        train=True,
        random_base=True,
        align_kps=ALIGNED_IDX,
        crop_kps=CROP_IDX,
        online=False,
        fsr_thresh=0.2,
    ):
        super().__init__(
            data_dir,
            transforms,
            selected_participants,
            split,
            sample_ratio,
            train,
            random_base,
            align_kps,
            crop_kps,
            online,
        )
        self.fsr_thresh = fsr_thresh

    @staticmethod
    def _motion_type_to_array(motion_type):
        mapping = {
            "force_pinch_index": np.array([1, 0, 0, 0, 1]),
            "force_pinch_middle": np.array([0, 1, 0, 0, 1]),
            "force_pinch_ring": np.array([0, 0, 1, 0, 1]),
            "force_fist_clench": np.array([1, 1, 1, 1, 1]),
            "surface_tap_all_force": np.array([1, 0, 0, 0, 0]),
            "surface_tap_index_force": np.array([1, 0, 0, 0, 0]),
        }
        if motion_type not in mapping:
            raise ValueError("Motion type is not valid in the mapping. continue")

        else:
            return mapping[motion_type]

    @staticmethod
    def _motion_type_to_label(motion_type):
        mapping = {
            "force_pinch_index": 1,
            "force_pinch_middle": 2,
            "force_pinch_ring": 3,
            "force_fist_clench": 4,
            "surface_tap_all_force": 5,
            "surface_tap_index_force": 6,
        }

        if motion_type not in mapping:
            raise ValueError("Motion type is not valid in the mapping. continue")

        else:
            return mapping[motion_type]

    def __getitem__(self, idx):
        frame = self.frames[idx]
        trial = self.trials_dict[frame["trial_id"]]
        motion_mapping = self._motion_type_to_array(trial["motion_type"])
        p_id = trial["p_id"]

        hamer_ann = np.load(
            self.frames_dir / frame["hamer_path"], allow_pickle=True
        ).item()

        orig_img_pil, base_img_pil, base_hamer_shape = self.load_base_imgs(
            trial, frame, p_id, hamer_ann
        )

        orig_img_pil = self.transforms(orig_img_pil)
        base_img_pil = self.transforms(base_img_pil)

        if frame["fsr_reading"] < self.fsr_thresh:
            tap_label = 0
        else:
            tap_label = self._motion_type_to_label(trial["motion_type"])

        ret = {
            "frame_id": frame["frame_id"],
            "p_id": p_id,
            "img": orig_img_pil,
            "base_img": base_img_pil,
            "force": frame["fsr_reading"] * motion_mapping,
            "tap_label": tap_label,
        }

        return ret


if __name__ == "__main__":
    from datasets.transforms import build_augment_transform

    # dataset = EgoDorsalBaseDataset(data_dir='/data/EgoDorsal/egodorsalpose', transforms=build_augment_transform(), selected_participants=[1,2,4,5,6,7,8,9,10,11], sample_ratio=1)
    dataset = EgoDorsalForceDataset(
        data_dir="/data/EgoDorsal/egodorsalforce",
        transforms=build_augment_transform(),
        selected_participants=[1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
        sample_ratio=1,
    )
