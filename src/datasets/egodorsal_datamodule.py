"""PyTorch Lightning data module for ego-dorsal datasets."""

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.egodorsal_dataset import EgoDorsalForceDataset, EgoDorsalPoseDataset
from datasets.transforms import build_augment_transform, build_dino_transform


class EgoDorsalDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        data_type,
        out_img_size: int = 512,
        batch_size: int = 64,
        random_base=True,
        leave_one_out=False,
        sample_ratio=1,
        all_participant_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        val_participants=[1],
        test_participants=[3],
        online=False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_type = data_type
        if self.data_type not in ("pose", "force"):
            raise ValueError("Data type not one of pose or force")

        if self.data_type == "pose":
            self.ds = EgoDorsalPoseDataset
        else:
            self.ds = EgoDorsalForceDataset

        self.base_transform = build_dino_transform(out_img_size)
        self.augment_transform = build_augment_transform()

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

        self.val_participants = val_participants
        self.test_participants = test_participants

        selected = self.val_participants + self.test_participants
        self.train_participants = [i for i in all_participant_ids if i not in selected]

        print(
            "train, test, val participants",
            self.train_participants,
            self.val_participants,
            self.test_participants,
        )

    def setup(self, stage=None):
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        test_transforms = T.Compose([T.ToTensor(), self.base_transform])
        train_transforms = T.Compose([self.augment_transform, self.base_transform])

        if not self.hparams.leave_one_out:
            if not self.data_train:
                self.data_train = self.ds(
                    self.hparams.data_dir,
                    split="train",
                    transforms=train_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )

            if not self.data_val:
                self.data_val = self.ds(
                    self.hparams.data_dir,
                    split="val",
                    transforms=train_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )

            if not self.data_test:
                self.data_test = self.ds(
                    self.hparams.data_dir,
                    split="test",
                    transforms=test_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )
        else:
            print(
                f"LEAVE ONE OUT ACTIVATED: Split: train: {self.train_participants}, val: {self.val_participants}, test: {self.test_participants}"
            )
            if not self.data_train:
                self.data_train = self.ds(
                    self.hparams.data_dir,
                    split="all",
                    selected_participants=self.train_participants,
                    transforms=train_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )

            if not self.data_val:
                self.data_val = self.ds(
                    self.hparams.data_dir,
                    split="all",
                    selected_participants=self.val_participants,
                    transforms=train_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )

            if not self.data_test:
                self.data_test = self.ds(
                    self.hparams.data_dir,
                    split="all",
                    selected_participants=self.test_participants,
                    transforms=test_transforms,
                    random_base=self.hparams.random_base,
                    sample_ratio=self.hparams.sample_ratio,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
