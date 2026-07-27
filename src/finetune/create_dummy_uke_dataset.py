"""Build a synthetic UKE-style dataset from SmartDoc15 images for pipeline smoke-testing.

Real UKE patient data cannot be shared, but the finetuning pipeline needs *some* data in
uke_train_dir/uke_val_dir (and a metadata CSV) to exercise the UKE code path, including the
patient-grouped k-fold split. This script repurposes SmartDoc15 images (which already have
ground-truth corner annotations) as stand-in "patients": it copies a sample of images into
uke_train_dir/uke_val_dir (and optionally a test dir), renaming them to embed a synthetic
pat_<N> id so extract_patient_group_from_path() has something real to group on, and writes a
matching metadata CSV with the original ground-truth corners carried over.

This is throwaway test data, not real training data - it exists purely to let you verify the
pipeline runs end-to-end before you plug in real annotated data.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

from finetuning import load_training_config

CORNER_COLS = ["tl_x", "tl_y", "bl_x", "bl_y", "br_x", "br_y", "tr_x", "tr_y"]


def load_valid_source_rows(metadata_path: Path, image_folder: Path) -> pd.DataFrame:
    """Load SmartDoc15 metadata rows that have full corner annotations and an existing image file."""
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=CORNER_COLS)
    exists_mask = df["image_path"].apply(lambda rel: (image_folder / rel).is_file())
    return df[exists_mask].reset_index(drop=True)


def sample_rows(df: pd.DataFrame, n: int, seed: int, used_indices: set) -> pd.DataFrame:
    """Sample n rows not already used, for reproducible non-overlapping splits."""
    available = df[~df.index.isin(used_indices)]
    if len(available) < n:
        raise ValueError(f"Not enough source images: need {n}, have {len(available)} left.")
    sampled = available.sample(n=n, random_state=seed)
    used_indices.update(sampled.index)
    return sampled


def assign_patient_groups(n_rows: int, n_patients: int, seed: int) -> list:
    """Round-robin-assign a shuffled list of pseudo-patient ids across n_rows items."""
    rng = random.Random(seed)
    groups = [f"pat_{i + 1}" for i in range(n_patients)]
    assignment = [groups[i % n_patients] for i in range(n_rows)]
    rng.shuffle(assignment)
    return assignment


def build_split(rows: pd.DataFrame, image_folder: Path, target_dir: Path, patient_ids: list, tag: str):
    """Copy sampled images into target_dir with pat_<n> filenames; return metadata records."""
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for (_, row), patient_id in zip(rows.iterrows(), patient_ids):
        src = image_folder / row["image_path"]
        stem = Path(row["image_path"]).with_suffix("").as_posix().replace("/", "_")
        new_name = f"{patient_id}__{tag}__{stem}{Path(row['image_path']).suffix}"
        dst = target_dir / new_name
        shutil.copy(src, dst)

        record = {"image_path": new_name}
        for col in CORNER_COLS + ["model_width", "model_height"]:
            record[col] = row[col]
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to training_config.json.")
    parser.add_argument("--train-count", type=int, default=75)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--test-dir", default=None, help="Where to copy held-out test images (defaults to a 'test' folder next to uke_train_dir). Point training_config.json's uke_test_dir at the same path so finetuning.py picks these images up.")
    parser.add_argument("--n-patients", type=int, default=10, help="Number of synthetic patient ids to spread train+val images across.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing metadata CSV / re-copy even if targets already have files.")
    args = parser.parse_args()

    config = load_training_config(args.config)

    metadata_path = Path(config["opensource_metadata_path"])
    image_folder = Path(config["opensource_image_folder_path"])
    uke_train_dir = Path(config["uke_train_dir"])
    uke_val_dir = Path(config["uke_val_dir"])
    uke_metadata_path = Path(config["uke_metadata_path"])
    test_dir = Path(args.test_dir) if args.test_dir else uke_train_dir.parent / "test"

    if any(d.exists() and any(d.iterdir()) for d in (uke_train_dir, uke_val_dir)) and not args.force:
        raise SystemExit(
            f"{uke_train_dir} or {uke_val_dir} already contains files. "
            "Pass --force to add more / regenerate."
        )

    print(f"Loading SmartDoc15 metadata from {metadata_path}")
    df = load_valid_source_rows(metadata_path, image_folder)
    print(f"{len(df)} SmartDoc15 images with valid ground truth available.")

    used_indices = set()
    train_rows = sample_rows(df, args.train_count, args.seed, used_indices)
    val_rows = sample_rows(df, args.val_count, args.seed + 1, used_indices)
    test_rows = sample_rows(df, args.test_count, args.seed + 2, used_indices)

    train_patients = assign_patient_groups(len(train_rows), args.n_patients, args.seed)
    val_patients = assign_patient_groups(len(val_rows), args.n_patients, args.seed + 1)
    test_patients = assign_patient_groups(len(test_rows), args.n_patients, args.seed + 100)

    print(f"Copying {len(train_rows)} images to {uke_train_dir}")
    train_records = build_split(train_rows, image_folder, uke_train_dir, train_patients, "train")
    print(f"Copying {len(val_rows)} images to {uke_val_dir}")
    val_records = build_split(val_rows, image_folder, uke_val_dir, val_patients, "val")
    print(f"Copying {len(test_rows)} images to {test_dir}")
    test_records = build_split(test_rows, image_folder, test_dir, test_patients, "test")

    all_records = train_records + val_records + test_records
    out_df = pd.DataFrame(all_records)

    uke_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if uke_metadata_path.is_file() and not args.force:
        raise SystemExit(f"{uke_metadata_path} already exists. Pass --force to overwrite.")
    out_df.to_csv(uke_metadata_path, index=False)

    print(f"\nWrote metadata for {len(out_df)} images to {uke_metadata_path}")
    print(f"Synthetic patients used: {args.n_patients} (train+val), separate ids for test.")
    print("This is dummy test data (real SmartDoc15 images relabeled as fake patients) - "
          "replace with real annotated data before drawing any real conclusions.")


if __name__ == "__main__":
    main()
