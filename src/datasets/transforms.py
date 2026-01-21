"""Data augmentation and preprocessing transforms."""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def build_augment_transform(
    brightness=0.10,
    contrast=0.10,
    gamma_jitter=0.10,
    noise_std=0.007,
    kernel_size=5,
    sigma=(0.1, 1.0),
):
    return T.Compose(
        [
            T.ColorJitter(brightness=brightness, contrast=contrast),
            T.ToTensor(),
            T.Lambda(
                lambda x: TF.adjust_gamma(
                    x, gamma=1.0 + random.uniform(-gamma_jitter, gamma_jitter), gain=1.0
                )
            ),
            T.Lambda(lambda x: (x + torch.randn_like(x) * noise_std).clamp(0.0, 1.0)),
            T.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        ]
    )


def build_dino_transform(resize_size):
    resize = T.Resize((resize_size, resize_size), antialias=True)
    normalize = T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return T.Compose([resize, normalize])
