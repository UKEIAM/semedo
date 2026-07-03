import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image, ImageOps


class DocumentDataset(Dataset):
    """Dataset for document segmentation with polygon-to-mask conversion and augmentations."""

    def __init__(
        self,
        image_paths,
        polygon_dict,
        augment=False,
        augmentation_profile="opensource",
        augment_prob=None,
        rotate_prob=None,
        brightness_prob=None,
        scale_prob=None,
        noise_prob=None,
        rotate_range=None,
        brightness_range=None,
        scale_range=None,
        noise_std=None,
    ):
        """Build the dataset; explicit kwargs override the augmentation_profile defaults."""
        self.image_paths = image_paths
        self.polygon_dict = polygon_dict
        self.augment = augment

        profile_defaults = {
            "opensource": {
                "augment_prob": 1.0,
                "rotate_prob": 1.0,
                "brightness_prob": 1.0,
                "scale_prob": 1.0,
                "noise_prob": 1.0,
                "rotate_range": (-90, 90),
                "brightness_range": (0.8, 1.2),
                "scale_range": (1.1, 2.0),
                "noise_std": 10.0,
            },
            "uke_light": {
                "augment_prob": 1.0,
                "rotate_prob": 1.0,
                "brightness_prob": 1.0,
                "scale_prob": 1.0,
                "noise_prob": 1.0,
                "rotate_range": (-90, 90),
                "brightness_range": (0.8, 1.2),
                "scale_range": (1.0, 1.2),
                "noise_std": 10.0,
            },
        }

        if augmentation_profile not in profile_defaults:
            raise ValueError(
                f"Unknown augmentation_profile='{augmentation_profile}'. "
                "Use 'opensource' or 'uke_light'."
            )

        defaults = profile_defaults[augmentation_profile]
        self.augment_prob = (
            defaults["augment_prob"] if augment_prob is None else augment_prob
        )
        self.rotate_prob = (
            defaults["rotate_prob"] if rotate_prob is None else rotate_prob
        )
        self.brightness_prob = (
            defaults["brightness_prob"] if brightness_prob is None else brightness_prob
        )
        self.scale_prob = defaults["scale_prob"] if scale_prob is None else scale_prob
        self.noise_prob = defaults["noise_prob"] if noise_prob is None else noise_prob
        self.rotate_range = (
            defaults["rotate_range"] if rotate_range is None else rotate_range
        )
        self.brightness_range = (
            defaults["brightness_range"]
            if brightness_range is None
            else brightness_range
        )
        self.scale_range = (
            defaults["scale_range"] if scale_range is None else scale_range
        )
        self.noise_std = defaults["noise_std"] if noise_std is None else noise_std

    def __len__(self):
        return len(self.image_paths)

    def polygon_to_mask(self, img_shape, polygon):
        """Rasterize a polygon into a binary mask matching image height/width."""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    def __getitem__(self, idx):
        """Load one sample, apply optional augmentations, and resize to model input size."""
        img_path = self.image_paths[idx]

        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        img = np.array(pil_img, dtype=np.float32) / 255.0

        polygon = self.polygon_dict[img_path]

        h, w = img.shape[:2]

        # --- Square padding ---
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        img_padded = np.pad(
            img,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        # Build the mask and pad it square as well
        gt_mask = self.polygon_to_mask(img.shape, polygon)
        gt_mask_padded = np.pad(
            gt_mask,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

        if self.augment and random.random() < self.augment_prob:
            if random.random() < self.rotate_prob:
                angle = random.uniform(self.rotate_range[0], self.rotate_range[1])
                rot_matrix = cv2.getRotationMatrix2D(
                    (max_dim / 2, max_dim / 2), angle, 1.0
                )
                img_padded = cv2.warpAffine(
                    img_padded,
                    rot_matrix,
                    (max_dim, max_dim),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                gt_mask_padded = cv2.warpAffine(
                    gt_mask_padded,
                    rot_matrix,
                    (max_dim, max_dim),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            if random.random() < self.brightness_prob:
                factor = random.uniform(
                    self.brightness_range[0], self.brightness_range[1]
                )
                img_padded = np.clip(img_padded * factor, 0, 1)

            if random.random() < self.scale_prob and self.scale_range[1] > 1.0:
                scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
                new_size = int(max_dim * scale_factor)
                img_scaled = cv2.resize(
                    img_padded, (new_size, new_size), interpolation=cv2.INTER_LINEAR
                )
                gt_mask_scaled = cv2.resize(
                    gt_mask_padded,
                    (new_size, new_size),
                    interpolation=cv2.INTER_NEAREST,
                )

                crop_x = max((new_size - max_dim) // 2, 0)
                crop_y = max((new_size - max_dim) // 2, 0)
                img_padded = img_scaled[
                    crop_y : crop_y + max_dim, crop_x : crop_x + max_dim
                ]
                gt_mask_padded = gt_mask_scaled[
                    crop_y : crop_y + max_dim, crop_x : crop_x + max_dim
                ]

            if random.random() < self.noise_prob and self.noise_std > 0:
                noise = np.random.normal(0, self.noise_std / 255.0, img_padded.shape)
                img_padded = np.clip(img_padded + noise, 0, 1)

        # --- Resize to 320x320 ---
        target_size = 320
        img_resized = cv2.resize(
            img_padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )
        gt_mask_resized = cv2.resize(
            gt_mask_padded, (target_size, target_size), interpolation=cv2.INTER_NEAREST
        )

        # Bring the mask into the expected (channel-first) shape
        gt_mask_resized = np.expand_dims(gt_mask_resized, axis=0).astype(np.float32)

        return {
            "image": torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1),
            "mask": torch.tensor(gt_mask_resized, dtype=torch.float32),
            "orig_size": (h, w),
            "orig_img": torch.tensor(img, dtype=torch.float32).permute(2, 0, 1),
            "orig_polygon": polygon,
            "image_path": img_path,
        }
