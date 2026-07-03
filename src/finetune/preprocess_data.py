import pandas as pd
import numpy as np
import os
import cv2
import glob
import random
import json


def iterate_though_folders(
    root_folder,
    train_paths,
    test_paths,
    val_paths,
    n_train_per_subfolder,
    n_test_per_subfolder,
    n_val_per_subfolder,
):
    """Builds train/test/val path lists by sampling images from each subfolder."""
    # all first-level subfolders (background domains)
    background_folders = [
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]

    for bg in background_folders:
        # all subfolders in this background
        subfolders = [
            os.path.join(bg, d)
            for d in os.listdir(bg)
            if os.path.isdir(os.path.join(bg, d))
        ]

        for sub in subfolders:
            # all images in this subfolder
            imgs = glob.glob(os.path.join(sub, "*.jpeg"))  # could be changed to *.png/*.jpg
            random.shuffle(imgs)  # shuffle randomly

            # split selection
            train_paths.extend(imgs[:n_train_per_subfolder])
            test_paths.extend(
                imgs[
                    n_train_per_subfolder : n_train_per_subfolder + n_test_per_subfolder
                ]
            )
            val_paths.extend(
                imgs[
                    n_train_per_subfolder + n_test_per_subfolder : n_train_per_subfolder
                    + n_test_per_subfolder
                    + n_val_per_subfolder
                ]
            )

    print(
        f"Train images: {len(train_paths)}, Test images: {len(test_paths)}, Val images: {len(val_paths)}"
    )
    return train_paths, test_paths, val_paths


def domain_aware_split(
    root_folder,
    backgrounds,
    n_train_per_subfolder=None,
    n_val_per_subfolder=None,
    n_test_per_subfolder=None,
):
    """Creates a background-aware train/val/test split with optional per-subfolder limits."""
    train_paths, val_paths, test_paths = [], [], []

    def collect(bg_list, n_limit_per_subfolder):
        """Collects image paths for a list of backgrounds with optional per-subfolder cap."""
        collected = []
        for bg in bg_list:
            bg_folder = os.path.join(root_folder, bg)
            subfolders = [
                os.path.join(bg_folder, d)
                for d in os.listdir(bg_folder)
                if os.path.isdir(os.path.join(bg_folder, d))
            ]
            for sub in subfolders:
                imgs = glob.glob(os.path.join(sub, "*.jpeg"))
                random.shuffle(imgs)
                # limit samples per subfolder
                if n_limit_per_subfolder is not None:
                    imgs = imgs[:n_limit_per_subfolder]
                collected += imgs
        return collected

    train_paths = collect(backgrounds["train"], n_train_per_subfolder)
    val_paths = collect(backgrounds["val"], n_val_per_subfolder)
    test_paths = collect(backgrounds["test"], n_test_per_subfolder)

    print("Split Summary:")
    print("Train:", len(train_paths))
    print("Val:", len(val_paths))
    print("Test:", len(test_paths))

    return train_paths, val_paths, test_paths


def _resolve_image_path(raw_path, image_folder_path):
    """Resolves a raw image path to an absolute path using the image root if needed."""
    if pd.isna(raw_path):
        return None

    raw_path = str(raw_path)
    if os.path.isabs(raw_path):
        return os.path.abspath(raw_path)
    return os.path.abspath(os.path.join(image_folder_path, raw_path))


def _extract_polygon_from_row(row):
    """Extracts polygon coordinates from a row using JSON points or legacy corner columns."""
    polygon_raw = row.get("polygon_points", None)

    # 1) New format: variable points stored as JSON
    if pd.notna(polygon_raw):
        try:
            loaded = json.loads(polygon_raw)
            polygon = []
            for point in loaded:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                x, y = point[0], point[1]
                if pd.notna(x) and pd.notna(y):
                    polygon.append([float(x), float(y)])

            if len(polygon) >= 3:
                return np.array(polygon, dtype=np.float32)
        except Exception:
            pass

    # 2) Fallback: legacy 4-corner format
    fallback_pairs = [
        ("tl_x", "tl_y"),
        ("bl_x", "bl_y"),
        ("br_x", "br_y"),
        ("tr_x", "tr_y"),
    ]
    polygon = []
    for x_col, y_col in fallback_pairs:
        if x_col in row and y_col in row:
            x_val, y_val = row[x_col], row[y_col]
            if pd.notna(x_val) and pd.notna(y_val):
                polygon.append([float(x_val), float(y_val)])

    if len(polygon) >= 3:
        return np.array(polygon, dtype=np.float32)

    return None


def get_gt_to_images(df, image_folder_path, train_paths, val_paths, test_paths):
    """Maps image paths to ground-truth polygons for train, val, and test splits."""
    polygon_train_dict = {}
    polygon_val_dict = {}
    polygon_test_dict = {}

    train_paths_set = {os.path.abspath(p) for p in train_paths}
    val_paths_set = {os.path.abspath(p) for p in val_paths}
    test_paths_set = {os.path.abspath(p) for p in test_paths}

    for _, row in df.iterrows():
        image_path = _resolve_image_path(row.get("image_path"), image_folder_path)
        if image_path is None:
            print("[WARNING] image_path missing, skipping...")
            continue

        if not os.path.isfile(image_path):
            print(f"[WARNING] Image does not exist: {image_path}, skipping...")
            continue

        gt_poly = _extract_polygon_from_row(row)
        if gt_poly is None:
            print(
                f"[WARNING] No valid polygon points for {image_path}, skipping..."
            )
            continue

        if image_path in train_paths_set:
            polygon_train_dict[image_path] = gt_poly
        elif image_path in val_paths_set:
            polygon_val_dict[image_path] = gt_poly
        elif image_path in test_paths_set:
            polygon_test_dict[image_path] = gt_poly
        else:
            print(f"[WARNING] Image {image_path} does not belong to any split")

    return polygon_train_dict, polygon_val_dict, polygon_test_dict
