import json
import torch
from u2net import U2NET
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import glob
from preprocess_data import get_gt_to_images
from evaluation_helpers import muti_bce_loss_fusion
import os
import random
import re
import hashlib
from dataloader import DocumentDataset
import wandb
from torch.utils.data import ConcatDataset, WeightedRandomSampler, SubsetRandomSampler
from pathlib import Path


DEFAULT_TRAINING_CONFIG_PATH = Path(__file__).resolve().parent / "training_config.json"


def _resolve_path_value(value, base_dir: Path):
    """Resolve a config path relative to the config file's directory if not absolute."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_training_config(config_path=None):
    """Load optional training config from JSON; fall back to defaults when missing."""
    candidate = (
        config_path
        or os.environ.get("SEMEDO_TRAINING_CONFIG")
        or DEFAULT_TRAINING_CONFIG_PATH
    )
    config_file = Path(candidate).expanduser()
    if not config_file.is_file():
        return {}
    with open(config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_dir = config_file.parent
    path_keys = [
        "model_path",
        "root_folder",
        "opensource_metadata_path",
        "opensource_image_folder_path",
        "uke_metadata_path",
        "uke_train_dir",
        "uke_val_dir",
        "uke_test_dir",
        "checkpoint_dir",
    ]
    for key in path_keys:
        if key in config and config[key] is not None:
            config[key] = _resolve_path_value(config[key], base_dir)

    return config


def seed_worker(worker_id):
    """Seed NumPy/Python RNGs per DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed: int):
    """Set deterministic seeds/configuration across Python, NumPy and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(model_path, device):
    """Load U2NET and pretrained weights."""
    net = U2NET(in_ch=3, out_ch=1)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    return net


def collect_paths_from_backgrounds(root_folder, background_list, n_limit_per_subfolder):
    """Collect image paths from selected background folders with optional per-subfolder limit."""
    collected = []
    for bg in sorted(background_list):
        bg_folder = os.path.join(root_folder, bg)
        subfolders = [
            os.path.join(bg_folder, d)
            for d in sorted(os.listdir(bg_folder))
            if os.path.isdir(os.path.join(bg_folder, d))
        ]
        for sub in subfolders:
            imgs = sorted(glob.glob(os.path.join(sub, "*.jpeg")))
            random.shuffle(imgs)
            if n_limit_per_subfolder is not None:
                imgs = imgs[:n_limit_per_subfolder]
            collected.extend(imgs)

    return collected


def compute_batch_metrics(preds, masks, thresh=0.5):
    """Compute mean segmentation metrics for a batch from thresholded predictions."""
    preds = (preds > thresh).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks - preds * masks).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    gt_area = masks.sum(dim=(1, 2, 3))

    iou = intersection / (union + 1e-8)
    precision = intersection / (pred_area + 1e-8)
    recall = intersection / (gt_area + 1e-8)
    dice = (2 * intersection) / (pred_area + gt_area + 1e-8)
    beta_sq = 4.0
    f2 = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall + 1e-8)

    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "dice": dice.mean().item(),
        "f2": f2.mean().item(),
    }


def evaluate_loader(net, loader, device):
    """Evaluate model on one loader and return loss + averaged metrics."""
    if loader is None:
        return None

    val_loss_sum = 0.0
    val_metrics_all = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(imgs)
            _, loss_sum = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, masks)

            val_loss_sum += loss_sum.item()
            val_metrics_all.append(compute_batch_metrics(d0, masks))
            n_batches += 1

    if n_batches == 0:
        return None

    metrics = {
        key: float(np.mean([item[key] for item in val_metrics_all]))
        for key in val_metrics_all[0]
    }
    metrics["loss"] = val_loss_sum / n_batches
    return metrics


def append_manifest_row(row, manifest_path):
    """Append one training-run row to the checkpoint manifest CSV, creating it if needed."""
    df_row = pd.DataFrame([row])
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        df_row.to_csv(manifest_file, index=False)
    else:
        df_row.to_csv(manifest_file, mode="a", header=False, index=False)


def build_checkpoint_path(checkpoint_dir: str, run_name: str, extension: str = ".pth"):
    """Build a checkpoint path that respects filesystem filename length limits."""
    raw_filename = f"{run_name}{extension}"

    try:
        max_name_bytes = os.pathconf(checkpoint_dir, "PC_NAME_MAX")
    except (AttributeError, OSError, ValueError):
        max_name_bytes = 255

    if len(raw_filename.encode("utf-8")) <= max_name_bytes:
        return os.path.join(checkpoint_dir, raw_filename)

    digest = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:12]
    tail = f"__{digest}{extension}"
    allowed_prefix_bytes = max_name_bytes - len(tail.encode("utf-8"))

    if allowed_prefix_bytes <= 0:
        return os.path.join(checkpoint_dir, f"{digest}{extension}")

    prefix = (
        run_name.encode("utf-8")[:allowed_prefix_bytes]
        .decode("utf-8", errors="ignore")
        .rstrip("_-")
    )
    if not prefix:
        return os.path.join(checkpoint_dir, f"{digest}{extension}")

    return os.path.join(checkpoint_dir, f"{prefix}{tail}")


def collate_train_val_batch(batch):
    """Collate only fixed-size tensors/fields needed in training and plotting."""
    return {
        "image": torch.stack([sample["image"] for sample in batch], dim=0),
        "mask": torch.stack([sample["mask"] for sample in batch], dim=0),
        "image_path": [sample["image_path"] for sample in batch],
    }


def collect_images_flat(folder_path):
    """Collect image files from one folder (non-recursive)."""
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(folder_path, pattern)))
    return sorted(paths)


def collect_images_from_folders(folder_paths):
    """Collect unique image files from multiple folders (non-recursive)."""
    all_paths = []
    for folder in folder_paths:
        if folder and os.path.isdir(folder):
            all_paths.extend(collect_images_flat(folder))
    return sorted({os.path.abspath(path) for path in all_paths})


def build_basename_index(paths):
    """Build basename -> absolute-paths index to support fallback lookup."""
    index = {}
    for path in paths:
        key = os.path.basename(path)
        index.setdefault(key, []).append(os.path.abspath(path))
    return index


def resolve_csv_image_path_multi_root(raw_path, candidate_roots, basename_index):
    """Resolve CSV image path against multiple roots with safe basename fallback."""
    if pd.isna(raw_path):
        return None, "missing"

    raw_path = str(raw_path).strip()
    if len(raw_path) == 0:
        return None, "missing"

    direct_candidates = []
    if os.path.isabs(raw_path):
        direct_candidates.append(os.path.abspath(raw_path))
    else:
        for root in candidate_roots:
            direct_candidates.append(os.path.abspath(os.path.join(root, raw_path)))

    existing = [path for path in direct_candidates if os.path.isfile(path)]
    if len(existing) == 1:
        return existing[0], "direct"
    if len(existing) > 1:
        return None, "ambiguous_direct"

    basename = os.path.basename(raw_path)
    basename_hits = basename_index.get(basename, [])
    if len(basename_hits) == 1:
        return basename_hits[0], "basename"
    if len(basename_hits) > 1:
        return None, "ambiguous_basename"

    return None, "not_found"


def normalize_uke_metadata_paths(df, candidate_roots, available_paths):
    """Normalize UKE CSV image paths to absolute paths across multiple roots."""
    basename_index = build_basename_index(available_paths)

    resolved_paths = []
    status_counts = {
        "direct": 0,
        "basename": 0,
        "missing": 0,
        "not_found": 0,
        "ambiguous_direct": 0,
        "ambiguous_basename": 0,
    }

    for raw_path in df["image_path"]:
        resolved, status = resolve_csv_image_path_multi_root(
            raw_path,
            candidate_roots,
            basename_index,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        resolved_paths.append(resolved)

    normalized = df.copy()
    normalized["image_path"] = resolved_paths
    normalized = normalized[normalized["image_path"].notna()].copy()

    print(
        "UKE metadata path resolution: "
        f"direct={status_counts['direct']} "
        f"basename={status_counts['basename']} "
        f"missing={status_counts['missing']} "
        f"not_found={status_counts['not_found']} "
        f"ambiguous_direct={status_counts['ambiguous_direct']} "
        f"ambiguous_basename={status_counts['ambiguous_basename']}"
    )

    return normalized


def extract_patient_group_from_path(image_path):
    """Extract patient group id (e.g. pat_12) from a file name."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # Supports patterns like Pat9, Pat_9, Pat-9, Inkedpat_21_1
    match = re.search(r"pat(?:[-_\s]?)([0-9]+)", stem, flags=re.IGNORECASE)
    if match:
        return f"pat_{int(match.group(1))}"

    fallback_tokens = re.split(r"[-_]", stem)
    if fallback_tokens:
        return fallback_tokens[0].lower()
    return stem.lower()


def build_patient_group_folds(image_paths, n_folds, seed):
    """Build patient-level folds so one patient never appears in both train/val."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    group_to_paths = {}
    for path in image_paths:
        group = extract_patient_group_from_path(path)
        group_to_paths.setdefault(group, []).append(path)

    groups = list(group_to_paths.keys())
    if len(groups) < n_folds:
        raise ValueError(
            f"Not enough patient groups ({len(groups)}) for {n_folds}-fold split."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups_sorted = sorted(
        groups, key=lambda group: len(group_to_paths[group]), reverse=True
    )

    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    for group in groups_sorted:
        best_fold = min(range(n_folds), key=lambda idx: fold_sizes[idx])
        fold_groups[best_fold].append(group)
        fold_sizes[best_fold] += len(group_to_paths[group])

    folds = []
    all_paths_set = set(image_paths)
    for fold_idx in range(n_folds):
        val_groups = set(fold_groups[fold_idx])
        val_paths = []
        for group in val_groups:
            val_paths.extend(group_to_paths[group])

        val_set = set(val_paths)
        train_paths = [path for path in image_paths if path not in val_set]
        if len(train_paths) == 0 or len(val_paths) == 0:
            raise ValueError(f"Invalid fold {fold_idx}: empty train or val set.")

        if (set(train_paths) | set(val_paths)) != all_paths_set:
            raise ValueError(
                "Fold split mismatch: train/val union does not match input."
            )

        folds.append(
            {
                "fold_idx": fold_idx + 1,
                "train_paths": sorted(train_paths),
                "val_paths": sorted(val_paths),
                "n_train_groups": len(
                    set(extract_patient_group_from_path(p) for p in train_paths)
                ),
                "n_val_groups": len(val_groups),
            }
        )

    return folds


def is_path_under_dir(path, directory):
    """Check whether an absolute path is located under directory (also resolved to absolute)."""
    try:
        common = os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)])
    except ValueError:
        return False
    return common == os.path.abspath(directory)


def build_uke_pool_and_test_holdout(training_config):
    """Load UKE metadata and split it into a k-fold pool (train+val dirs) and a held-out
    test set (uke_test_dir) that no fold model ever trains or validates on.
    """
    uke_metadata_path = training_config.get(
        "uke_metadata_path",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/train_uke_model_coordinates.csv",
    )
    uke_train_dir = training_config.get(
        "uke_train_dir",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/train",
    )
    uke_val_dir = training_config.get(
        "uke_val_dir", "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/val"
    )
    uke_test_dir = training_config.get(
        "uke_test_dir",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/test",
    )
    uke_data_dirs = [uke_train_dir, uke_val_dir, uke_test_dir]

    if not os.path.isfile(uke_metadata_path):
        raise FileNotFoundError(
            f"include_uke_in_training is true but uke_metadata_path='{uke_metadata_path}' "
            "does not exist. Either point uke_metadata_path/uke_train_dir/uke_val_dir/"
            "uke_test_dir in training_config.json at your own institutional data (see the "
            "annotation tool to create it), or set include_uke_in_training to false to train "
            "on SmartDoc15 alone."
        )

    df_uke_raw = pd.read_csv(uke_metadata_path)
    uke_available_paths = collect_images_from_folders(uke_data_dirs)
    df_uke = normalize_uke_metadata_paths(
        df_uke_raw,
        candidate_roots=uke_data_dirs,
        available_paths=uke_available_paths,
    )

    uke_paths_from_csv = sorted(
        {
            os.path.abspath(path)
            for path in df_uke["image_path"].tolist()
            if isinstance(path, str)
        }
    )
    if len(uke_paths_from_csv) == 0:
        raise ValueError("No usable UKE image paths found after metadata normalization.")

    test_paths = sorted(
        path for path in uke_paths_from_csv if is_path_under_dir(path, uke_test_dir)
    )
    test_paths_set = set(test_paths)
    pool_paths = sorted(path for path in uke_paths_from_csv if path not in test_paths_set)

    df_test_filtered = filter_df_to_selected_paths(df_uke, "/", test_paths)
    _, _, polygon_test = get_gt_to_images(df_test_filtered, "/", [], [], test_paths)
    test_paths_with_gt = sorted(path for path in test_paths if path in polygon_test)

    print(
        "UKE holdout reserved from k-fold pool: "
        f"pool={len(pool_paths)} test={len(test_paths)} test_with_gt={len(test_paths_with_gt)}"
    )

    return {
        "df_uke": df_uke,
        "pool_paths": pool_paths,
        "test_paths": test_paths_with_gt,
        "polygon_test": polygon_test,
    }


def sample_paths_exact(paths, n_samples):
    """Sample exactly n_samples unique items from paths."""
    if n_samples is None:
        return list(paths)
    if n_samples > len(paths):
        raise ValueError(
            f"Requested {n_samples} samples, but only {len(paths)} available."
        )
    return random.sample(list(paths), n_samples)


def resolve_csv_image_path(raw_path, image_folder_path):
    """Resolve absolute/relative image paths from metadata rows."""
    if pd.isna(raw_path):
        return None

    raw_path = str(raw_path)
    if os.path.isabs(raw_path):
        return os.path.abspath(raw_path)
    return os.path.abspath(os.path.join(image_folder_path, raw_path))


def filter_df_to_selected_paths(df, image_folder_path, selected_paths):
    """Filter metadata to rows that belong to selected image paths only."""
    selected_set = {os.path.abspath(path) for path in selected_paths}
    if len(selected_set) == 0:
        return df.iloc[0:0].copy()

    resolved = df["image_path"].apply(
        lambda value: resolve_csv_image_path(value, image_folder_path)
    )
    keep_mask = resolved.isin(selected_set)
    return df[keep_mask].copy()


def build_source_train_datasets(paths, polygon_dict, aug_per_img, augmentation_profile):
    """Build one source-specific train dataset list (base + augmented copies)."""
    if len(paths) == 0:
        return []

    datasets = [
        DocumentDataset(
            paths,
            polygon_dict,
            augment=False,
            augmentation_profile=augmentation_profile,
        )
    ]
    for _ in range(aug_per_img):
        datasets.append(
            DocumentDataset(
                paths,
                polygon_dict,
                augment=True,
                augmentation_profile=augmentation_profile,
            )
        )
    return datasets


def build_train_dataset_and_sampler(
    source_datasets,
    balance_domains=True,
    allow_duplicate_samples_per_epoch=False,
):
    """Create train dataset and optional domain-balanced sampler."""
    ordered_parts = []
    source_sizes = {}

    for source_name, datasets in source_datasets.items():
        if not datasets:
            continue
        source_size = int(sum(len(ds) for ds in datasets))
        if source_size == 0:
            continue
        source_sizes[source_name] = source_size
        for ds in datasets:
            ordered_parts.append((source_name, ds))

    if not ordered_parts:
        raise ValueError("No training datasets available to build train loader.")

    if len(ordered_parts) == 1:
        return ordered_parts[0][1], None, source_sizes

    concat_dataset = ConcatDataset([ds for _, ds in ordered_parts])

    if not balance_domains or len(source_sizes) <= 1:
        return concat_dataset, None, source_sizes

    source_names = list(source_sizes.keys())
    per_source_mass = 1.0 / len(source_names)
    per_sample_weight = {
        source_name: per_source_mass / source_sizes[source_name]
        for source_name in source_names
    }

    sample_weights = []
    for source_name, ds in ordered_parts:
        sample_weights.extend([per_sample_weight[source_name]] * len(ds))

    if allow_duplicate_samples_per_epoch:
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return concat_dataset, sampler, source_sizes

    source_to_indices = {source_name: [] for source_name in source_names}
    offset = 0
    for source_name, ds in ordered_parts:
        ds_len = len(ds)
        source_to_indices[source_name].extend(range(offset, offset + ds_len))
        offset += ds_len

    target_per_source = min(len(indices) for indices in source_to_indices.values())
    sampled_indices = []
    for source_name in source_names:
        source_indices = source_to_indices[source_name]
        if len(source_indices) > target_per_source:
            sampled_indices.extend(random.sample(source_indices, target_per_source))
        else:
            sampled_indices.extend(source_indices)

    random.shuffle(sampled_indices)
    sampler = SubsetRandomSampler(sampled_indices)

    balanced_sizes = {source_name: target_per_source for source_name in source_names}
    return concat_dataset, sampler, balanced_sizes


def main():
    """Entry point for the full OpenSource/UKE training sweep."""
    training_config = load_training_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    set_global_seed(seed)

    base_run_name = training_config.get("base_run_name", "uke_ratio_sweep")
    final_aug_by_scenario = training_config.get(
        "final_aug_by_scenario",
        {
            "ratio_20_80": 5,
            "ratio_20_80_val_uke_only": 4,
            "ratio_30_70": 5,
            "ratio_30_70_val_uke_only": 5,
            "ratio_50_50": 5,
            "ratio_50_50_val_uke_only": 6,
            "uke_only_train_val_uke": 4,
            "opensource_only": 5,
        },
    )

    model_path = training_config.get(
        "model_path", "/workspaces/de.uke.iam.soos-document-ai-lab/u2net.pth"
    )

    root_folder = training_config.get(
        "root_folder",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_models/frames_orig",
    )
    opensource_metadata_path = training_config.get(
        "opensource_metadata_path",
        "/workspaces/de.uke.iam.soos-document-ai-lab/eval_models/frames_metadata.csv",
    )
    opensource_image_folder_path = training_config.get(
        "opensource_image_folder_path",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_models/frames_orig",
    )

    include_uke_in_training = bool(training_config.get("include_uke_in_training", True))
    balance_train_domains = bool(training_config.get("balance_train_domains", False))
    allow_duplicate_samples_per_epoch = bool(
        training_config.get("allow_duplicate_samples_per_epoch", False)
    )
    training_scenario = training_config.get("training_scenario", "fixed_ratio_counts")

    wandb_project = training_config.get("wandb_project", "uke_u2net_cv")
    wandb_entity = training_config.get("wandb_entity")
    wandb_mode = training_config.get("wandb_mode", "disabled")

    ratio_scenarios = training_config.get(
        "ratio_scenarios",
        [
            {
                "name": "ratio_50_50",
                "uke_ratio": 0.5,
                "opensource_ratio": 0.5,
                "opensource_train_orig": 75,
                "opensource_val_orig": 20,
                "include_opensource_in_train": True,
                "include_opensource_in_val": True,
            },
            {
                "name": "ratio_30_70",
                "uke_ratio": 0.3,
                "opensource_ratio": 0.7,
                "opensource_train_orig": 175,
                "opensource_val_orig": 47,
                "include_opensource_in_train": True,
                "include_opensource_in_val": True,
            },
            {
                "name": "ratio_20_80",
                "uke_ratio": 0.2,
                "opensource_ratio": 0.8,
                "opensource_train_orig": 300,
                "opensource_val_orig": 80,
                "include_opensource_in_train": True,
                "include_opensource_in_val": True,
            },
            {
                "name": "ratio_50_50_val_uke_only",
                "uke_ratio": 0.5,
                "opensource_ratio": 0.5,
                "opensource_train_orig": 75,
                "opensource_val_orig": 0,
                "include_opensource_in_train": True,
                "include_opensource_in_val": False,
            },
            {
                "name": "ratio_30_70_val_uke_only",
                "uke_ratio": 0.3,
                "opensource_ratio": 0.7,
                "opensource_train_orig": 175,
                "opensource_val_orig": 0,
                "include_opensource_in_train": True,
                "include_opensource_in_val": False,
            },
            {
                "name": "ratio_20_80_val_uke_only",
                "uke_ratio": 0.2,
                "opensource_ratio": 0.8,
                "opensource_train_orig": 300,
                "opensource_val_orig": 0,
                "include_opensource_in_train": True,
                "include_opensource_in_val": False,
            },
            {
                "name": "uke_only_train_val_uke",
                "uke_ratio": 1.0,
                "opensource_ratio": 0.0,
                "opensource_train_orig": 0,
                "opensource_val_orig": 0,
                "include_opensource_in_train": False,
                "include_opensource_in_val": False,
            },
        ],
    )
    run_all_ratio_scenarios = bool(
        training_config.get("run_all_ratio_scenarios", False)
    )
    selected_ratio_name = training_config.get("selected_ratio_name")

    if run_all_ratio_scenarios:
        active_ratio_scenarios = list(ratio_scenarios)
    else:
        if selected_ratio_name:
            active_ratio_scenarios = [
                ratio_cfg
                for ratio_cfg in ratio_scenarios
                if ratio_cfg.get("name") == selected_ratio_name
            ]
            if not active_ratio_scenarios:
                available_names = [cfg.get("name") for cfg in ratio_scenarios]
                raise ValueError(
                    f"selected_ratio_name='{selected_ratio_name}' not found in ratio_scenarios. "
                    f"Available: {available_names}"
                )
        else:
            active_ratio_scenarios = [ratio_scenarios[0]] if ratio_scenarios else []

    if not active_ratio_scenarios:
        raise ValueError("No active ratio scenario configured.")

    use_patient_group_kfold_for_uke = bool(
        training_config.get("use_patient_group_kfold_for_uke", True)
    )
    uke_n_folds = int(training_config.get("uke_n_folds", 5))
    active_uke_fold_index = int(training_config.get("active_uke_fold_index", 0))
    run_all_uke_folds = bool(training_config.get("run_all_uke_folds", True))

    uke_train_limit = training_config.get("uke_train_limit", None)
    uke_val_limit = training_config.get("uke_val_limit", None)

    checkpoint_dir = training_config.get(
        "checkpoint_dir",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/uke_models",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    manifest_path = os.path.join(checkpoint_dir, "training_runs_manifest.csv")

    backgrounds = training_config.get(
        "backgrounds",
        {
            "train": ["background02", "background05"],  # harder backgrounds
            "val": ["background04"],  # medium difficulty
            "test": ["background01", "background03"],  # easier cases
        },
    )

    # Learning-rate sweep list
    lr_list = training_config.get("lr_list", [1e-3])

    # Number of OpenSource test images per subfolder (fixed for all scenarios)
    n_test_per_subfolder = int(training_config.get("n_test_per_subfolder", 12))

    # Optional explicit model preload (kept as reference)
    # net = load_model(model_path, device)

    df_opensource = pd.read_csv(opensource_metadata_path)

    uke_train_paths_all = []
    uke_val_paths_all = []
    uke_fold_contexts = [
        {
            "fold_idx": None,
            "n_folds": 1,
            "train_paths": [],
            "val_paths": [],
            "n_train_groups": None,
            "n_val_groups": None,
        }
    ]
    df_uke = None
    uke_test_paths_all = []
    if include_uke_in_training:
        uke_holdout = build_uke_pool_and_test_holdout(training_config)
        df_uke = uke_holdout["df_uke"]
        uke_paths_from_csv = uke_holdout["pool_paths"]
        uke_test_paths_all = uke_holdout["test_paths"]

        if use_patient_group_kfold_for_uke:
            uke_folds = build_patient_group_folds(
                image_paths=uke_paths_from_csv,
                n_folds=uke_n_folds,
                seed=seed,
            )

            if run_all_uke_folds:
                uke_fold_contexts = [
                    {
                        "fold_idx": fold["fold_idx"],
                        "n_folds": uke_n_folds,
                        "train_paths": fold["train_paths"],
                        "val_paths": fold["val_paths"],
                        "n_train_groups": fold["n_train_groups"],
                        "n_val_groups": fold["n_val_groups"],
                    }
                    for fold in uke_folds
                ]
                print(f"UKE patient-group CV enabled: running all {uke_n_folds} folds.")
            else:
                if not (0 <= active_uke_fold_index < len(uke_folds)):
                    raise ValueError(
                        f"active_uke_fold_index={active_uke_fold_index} out of range [0, {len(uke_folds) - 1}]"
                    )

                active_fold = uke_folds[active_uke_fold_index]
                uke_fold_contexts = [
                    {
                        "fold_idx": active_fold["fold_idx"],
                        "n_folds": uke_n_folds,
                        "train_paths": active_fold["train_paths"],
                        "val_paths": active_fold["val_paths"],
                        "n_train_groups": active_fold["n_train_groups"],
                        "n_val_groups": active_fold["n_val_groups"],
                    }
                ]

                print(
                    "UKE patient-group fold selected: "
                    f"fold={active_fold['fold_idx']}/{uke_n_folds} "
                    f"train_imgs={len(active_fold['train_paths'])} "
                    f"val_imgs={len(active_fold['val_paths'])} "
                    f"train_groups={active_fold['n_train_groups']} "
                    f"val_groups={active_fold['n_val_groups']}"
                )
        else:
            if uke_train_limit is not None:
                uke_train_paths_all = uke_paths_from_csv[:uke_train_limit]
                val_start = uke_train_limit
            else:
                val_start = 0
                uke_train_paths_all = list(uke_paths_from_csv)

            if uke_val_limit is not None:
                uke_val_paths_all = uke_paths_from_csv[
                    val_start : val_start + uke_val_limit
                ]
            else:
                uke_val_paths_all = []

            print(
                "UKE fallback split selected: "
                f"train={len(uke_train_paths_all)} val={len(uke_val_paths_all)}"
            )

            uke_fold_contexts = [
                {
                    "fold_idx": None,
                    "n_folds": 1,
                    "train_paths": list(uke_train_paths_all),
                    "val_paths": list(uke_val_paths_all),
                    "n_train_groups": None,
                    "n_val_groups": None,
                }
            ]

    # Optional explicit data preparation (kept as reference)
    # train_loader, val_loader, test_loader = prepare_data(
    #     root_folder,
    #     metadata_path,
    #     image_folder_path,
    #     n_train_per_subfolder,
    #     n_test_per_subfolder,
    #     n_val_per_subfolder,
    # )

    # Training parameters
    epochs = int(training_config.get("epochs", 20))
    patience = int(training_config.get("patience", 6))

    for fold_ctx in uke_fold_contexts:
        uke_train_paths_all = list(fold_ctx["train_paths"])
        uke_val_paths_all = list(fold_ctx["val_paths"])
        fold_idx = fold_ctx["fold_idx"]
        n_folds = fold_ctx["n_folds"]

        fold_tag = (
            f"fold{fold_idx}of{n_folds}" if fold_idx is not None else "single_split"
        )

        print(
            f"\n===== UKE SPLIT {fold_tag} | "
            f"train_imgs={len(uke_train_paths_all)} val_imgs={len(uke_val_paths_all)} "
            f"train_groups={fold_ctx['n_train_groups']} val_groups={fold_ctx['n_val_groups']} "
            f"held_out_test_imgs={len(uke_test_paths_all)} =====\n"
        )

        for ratio_cfg in active_ratio_scenarios:
            scenario_name = ratio_cfg["name"]
            if scenario_name not in final_aug_by_scenario:
                raise ValueError(
                    f"Missing final augmentation setting for scenario '{scenario_name}'."
                )
            scenario_aug_per_img = final_aug_by_scenario[scenario_name]

            if ratio_cfg.get("match_uke_counts"):
                # Treat the actual UKE folder count as fixed and scale the OpenSource
                # count to hit the configured uke_ratio/opensource_ratio split.
                uke_ratio = ratio_cfg["uke_ratio"]
                opensource_ratio = ratio_cfg["opensource_ratio"]
                if uke_ratio <= 0:
                    raise ValueError(
                        f"match_uke_counts requires uke_ratio > 0 in scenario '{scenario_name}'."
                    )
                os_per_uke = opensource_ratio / uke_ratio
                opensource_train_orig = round(len(uke_train_paths_all) * os_per_uke)
                opensource_val_orig = round(len(uke_val_paths_all) * os_per_uke)
            else:
                opensource_train_orig = ratio_cfg["opensource_train_orig"]
                opensource_val_orig = ratio_cfg["opensource_val_orig"]
            include_opensource_in_train = ratio_cfg.get(
                "include_opensource_in_train", True
            )
            include_opensource_in_val = ratio_cfg.get("include_opensource_in_val", True)

            print(
                f"\n===== SCENARIO {scenario_name} | "
                f"OS train={opensource_train_orig} val={opensource_val_orig} | "
                f"UKE train={len(uke_train_paths_all)} val={len(uke_val_paths_all)} | "
                f"aug_per_img={scenario_aug_per_img} | split={fold_tag} =====\n"
            )

            run_seed = seed
            aug_per_img = scenario_aug_per_img
            set_global_seed(run_seed)

            set_global_seed(run_seed + 99_999)
            test_paths = collect_paths_from_backgrounds(
                root_folder,
                backgrounds["test"],
                n_test_per_subfolder,
            )

            set_global_seed(run_seed)

            train_pool = collect_paths_from_backgrounds(
                root_folder,
                backgrounds["train"],
                n_limit_per_subfolder=None,
            )
            val_pool = collect_paths_from_backgrounds(
                root_folder,
                backgrounds["val"],
                n_limit_per_subfolder=None,
            )

            train_paths = (
                sample_paths_exact(train_pool, opensource_train_orig)
                if include_opensource_in_train
                else []
            )
            val_paths = (
                sample_paths_exact(val_pool, opensource_val_orig)
                if include_opensource_in_val
                else []
            )

            print(
                f"Opensource selected: train={len(train_paths)} val={len(val_paths)}"
            )

            df_open_filtered = filter_df_to_selected_paths(
                df_opensource,
                opensource_image_folder_path,
                train_paths + val_paths + test_paths,
            )
            polygon_train_open, polygon_val_open, _ = get_gt_to_images(
                df_open_filtered,
                opensource_image_folder_path,
                train_paths,
                val_paths,
                test_paths,
            )

            train_paths_open = [
                path for path in train_paths if path in polygon_train_open
            ]
            val_paths_open = [
                path for path in val_paths if path in polygon_val_open
            ]

            source_train_datasets = {}
            if include_opensource_in_train:
                source_train_datasets["opensource"] = build_source_train_datasets(
                    train_paths_open,
                    polygon_train_open,
                    aug_per_img=aug_per_img,
                    augmentation_profile="opensource",
                )

            val_dataset_parts = []
            if include_opensource_in_val and len(val_paths_open) > 0:
                val_dataset_parts.append(
                    DocumentDataset(
                        val_paths_open,
                        polygon_val_open,
                        augment=False,
                        augmentation_profile="opensource",
                    )
                )

            uke_train_with_gt = 0
            uke_val_with_gt = 0
            train_paths_uke = []
            val_paths_uke = []
            polygon_val_uke = {}
            if include_uke_in_training and df_uke is not None:
                df_uke_filtered = filter_df_to_selected_paths(
                    df_uke,
                    "/",
                    uke_train_paths_all + uke_val_paths_all,
                )
                polygon_train_uke, polygon_val_uke, _ = get_gt_to_images(
                    df_uke_filtered,
                    "/",
                    uke_train_paths_all,
                    uke_val_paths_all,
                    [],
                )

                train_paths_uke = [
                    path for path in uke_train_paths_all if path in polygon_train_uke
                ]
                val_paths_uke = [
                    path for path in uke_val_paths_all if path in polygon_val_uke
                ]

                uke_train_with_gt = len(train_paths_uke)
                uke_val_with_gt = len(val_paths_uke)

                source_train_datasets["uke"] = build_source_train_datasets(
                    train_paths_uke,
                    polygon_train_uke,
                    aug_per_img=aug_per_img,
                    augmentation_profile="uke_light",
                )

                if len(val_paths_uke) > 0:
                    val_dataset_parts.append(
                        DocumentDataset(
                            val_paths_uke,
                            polygon_val_uke,
                            augment=False,
                            augmentation_profile="uke_light",
                        )
                    )

            train_dataset, train_sampler, train_source_sizes = (
                build_train_dataset_and_sampler(
                    source_train_datasets,
                    balance_domains=balance_train_domains,
                    allow_duplicate_samples_per_epoch=allow_duplicate_samples_per_epoch,
                )
            )

            if len(val_dataset_parts) == 0:
                raise ValueError(
                    f"No validation datasets available in scenario '{scenario_name}'."
                )
            if len(val_dataset_parts) == 1:
                val_dataset = val_dataset_parts[0]
            else:
                val_dataset = ConcatDataset(val_dataset_parts)

            print(
                "Train source sizes (after augmentation): "
                + ", ".join(
                    f"{name}={count}" for name, count in train_source_sizes.items()
                )
            )
            print(
                f"Val sizes: opensource={len(val_paths_open)} uke={uke_val_with_gt} | "
                f"balanced_sampler={train_sampler is not None}"
            )

            data_loader_generator = torch.Generator()
            data_loader_generator.manual_seed(run_seed + 1)

            train_loader = DataLoader(
                train_dataset,
                batch_size=12,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                worker_init_fn=seed_worker,
                generator=data_loader_generator,
                collate_fn=collate_train_val_batch,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                collate_fn=collate_train_val_batch,
            )

            val_loader_opensource = None
            if include_opensource_in_val and len(val_paths_open) > 0:
                val_loader_opensource = DataLoader(
                    DocumentDataset(
                        val_paths_open,
                        polygon_val_open,
                        augment=False,
                        augmentation_profile="opensource",
                    ),
                    batch_size=1,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    collate_fn=collate_train_val_batch,
                )

            val_loader_uke = None
            if uke_val_with_gt > 0:
                val_loader_uke = DataLoader(
                    DocumentDataset(
                        val_paths_uke,
                        polygon_val_uke,
                        augment=False,
                        augmentation_profile="uke_light",
                    ),
                    batch_size=1,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    collate_fn=collate_train_val_batch,
                )

            for lr in lr_list:
                print(
                    f"\n\n===== START TRAINING LR = {lr} | scenario = {scenario_name} | "
                    f"augs_per_img = {aug_per_img} =====\n"
                )

                # Reload model for a fair comparison between runs.
                net = load_model(model_path, device)

                optimizer = torch.optim.Adam(
                    net.parameters(),
                    lr=lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                )

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=3
                )

                best_score = -float("inf")
                best_val_metrics = None
                patience_counter = 0
                global_step = 0

                train_domain_tag = (
                    "mixed" if include_opensource_in_train else "ukeonly"
                )
                val_domain_tag = "mixed" if include_opensource_in_val else "ukeonly"
                run_name = (
                    f"{base_run_name}"
                    f"__scn-{scenario_name}"
                    f"__fold-{fold_tag}"
                    f"__train-{train_domain_tag}"
                    f"__val-{val_domain_tag}"
                    f"__aug-{aug_per_img}"
                    f"__lr-{lr}"
                    f"__seed-{run_seed}"
                )
                checkpoint_path = build_checkpoint_path(checkpoint_dir, run_name)
                checkpoint_filename = os.path.basename(checkpoint_path)
                if checkpoint_filename != f"{run_name}.pth":
                    print(
                        "Checkpoint filename shortened for filesystem compatibility: "
                        f"{checkpoint_filename}"
                    )
                wandb.init(
                    entity=wandb_entity,
                    project=wandb_project,
                    mode=wandb_mode,
                    group=base_run_name,
                    name=run_name,
                    reinit=True,
                    config={
                        "lr": lr,
                        "batch": 12,
                        "epochs": epochs,
                        "loss": "multi_bce",
                        "augs_per_img": aug_per_img,
                        "training_scenario": training_scenario,
                        "ratio_name": scenario_name,
                        "uke_ratio": ratio_cfg["uke_ratio"],
                        "opensource_ratio": ratio_cfg["opensource_ratio"],
                        "include_opensource_in_train": include_opensource_in_train,
                        "include_opensource_in_val": include_opensource_in_val,
                        "opensource_train_orig_target": opensource_train_orig,
                        "opensource_val_orig_target": opensource_val_orig,
                        "include_uke_in_training": include_uke_in_training,
                        "balance_train_domains": balance_train_domains,
                        "allow_duplicate_samples_per_epoch": allow_duplicate_samples_per_epoch,
                        "uke_train_selected": len(uke_train_paths_all),
                        "uke_val_selected": len(uke_val_paths_all),
                        "uke_train_with_gt": uke_train_with_gt,
                        "uke_val_with_gt": uke_val_with_gt,
                        "opensource_train_with_gt": len(train_paths_open),
                        "opensource_val_with_gt": len(val_paths_open),
                        "seed": run_seed,
                        "uke_fold_tag": fold_tag,
                        "uke_fold_idx": fold_idx,
                        "uke_n_folds": n_folds,
                        "uke_train_groups": fold_ctx["n_train_groups"],
                        "uke_val_groups": fold_ctx["n_val_groups"],
                        "uke_test_held_out": len(uke_test_paths_all),
                    },
                )

                # ================= TRAIN LOOP =================
                for epoch in range(epochs):
                    net.train()
                    epoch_loss = 0

                    for batch in train_loader:
                        imgs = batch["image"].to(device)
                        masks = batch["mask"].to(device)

                        optimizer.zero_grad()
                        d0, d1, d2, d3, d4, d5, d6 = net(imgs)

                        loss0, loss = muti_bce_loss_fusion(
                            d0, d1, d2, d3, d4, d5, d6, masks
                        )

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                        metrics = compute_batch_metrics(d0.detach(), masks)

                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/precision": metrics["precision"],
                                "train/recall": metrics["recall"],
                                "train/dice": metrics["dice"],
                                "train/f2": metrics["f2"],
                                "train/iou": metrics["iou"],
                                "lr": optimizer.param_groups[0]["lr"],
                                "step": global_step,
                                "augs_per_img": aug_per_img,
                            }
                        )

                        global_step += 1

                    train_loss = epoch_loss / len(train_loader)

                    # ================= VALIDATION =================
                    net.eval()

                    val_metrics = evaluate_loader(net, val_loader, device)
                    val_metrics_uke = evaluate_loader(net, val_loader_uke, device)
                    val_metrics_opensource = evaluate_loader(
                        net, val_loader_opensource, device
                    )

                    val_loss = val_metrics["loss"]

                    if val_metrics_uke is not None:
                        selection_f2 = val_metrics_uke["f2"]
                        selection_dice = val_metrics_uke["dice"]
                        scheduler_target_loss = val_metrics_uke["loss"]
                    else:
                        selection_f2 = val_metrics["f2"]
                        selection_dice = val_metrics["dice"]
                        scheduler_target_loss = val_metrics["loss"]

                    selection_score = 0.7 * selection_f2 + 0.3 * selection_dice

                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_epoch_loss": train_loss,
                            "val_loss": val_loss,
                            "val_precision": val_metrics["precision"],
                            "val_recall": val_metrics["recall"],
                            "val_dice": val_metrics["dice"],
                            "val_f2": val_metrics["f2"],
                            "val_iou": val_metrics["iou"],
                            "val_selection_score": selection_score,
                            "augs_per_img": aug_per_img,
                        }
                    )

                    if val_metrics_uke is not None:
                        wandb.log(
                            {
                                "val_uke_loss": val_metrics_uke["loss"],
                                "val_uke_precision": val_metrics_uke["precision"],
                                "val_uke_recall": val_metrics_uke["recall"],
                                "val_uke_dice": val_metrics_uke["dice"],
                                "val_uke_f2": val_metrics_uke["f2"],
                                "val_uke_iou": val_metrics_uke["iou"],
                            }
                        )

                    if val_metrics_opensource is not None:
                        wandb.log(
                            {
                                "val_opensource_loss": val_metrics_opensource["loss"],
                                "val_opensource_precision": val_metrics_opensource[
                                    "precision"
                                ],
                                "val_opensource_recall": val_metrics_opensource[
                                    "recall"
                                ],
                                "val_opensource_dice": val_metrics_opensource["dice"],
                                "val_opensource_f2": val_metrics_opensource["f2"],
                                "val_opensource_iou": val_metrics_opensource["iou"],
                            }
                        )

                    scheduler.step(scheduler_target_loss)

                    # Early stopping (maximize UKE-focused selection score)
                    if selection_score > best_score:
                        best_score = selection_score
                        best_val_metrics = (
                            val_metrics_uke.copy()
                            if val_metrics_uke is not None
                            else val_metrics.copy()
                        )
                        best_val_metrics["selection_score"] = selection_score
                        patience_counter = 0

                        torch.save(net.state_dict(), checkpoint_path)

                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print("Early stopping")
                            break

                wandb.finish()

                manifest_row = {
                    "base_run_name": base_run_name,
                    "run_name": run_name,
                    "scenario_name": scenario_name,
                    "fold_idx": fold_idx,
                    "fold_tag": fold_tag,
                    "n_folds": n_folds,
                    "lr": lr,
                    "aug_per_img": aug_per_img,
                    "seed": run_seed,
                    "train_domain_tag": train_domain_tag,
                    "val_domain_tag": val_domain_tag,
                    "checkpoint_path": checkpoint_path,
                    "uke_test_held_out": len(uke_test_paths_all),
                }
                manifest_row.update(
                    {f"best_val_{key}": value for key, value in best_val_metrics.items()}
                )
                try:
                    append_manifest_row(manifest_row, manifest_path)
                except Exception as e:
                    print("Failed to append checkpoint manifest row:", e)

    print("\n===== TRAINING RUNS DONE =====")


if __name__ == "__main__":
    main()
