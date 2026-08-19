"""Evaluate fold-ensembles (and the pretrained baseline) on the held-out UKE test set.

Reads the checkpoint manifest written by finetuning.py, groups fold checkpoints that
belong to the same (scenario, lr, aug, seed, domain) run, averages their probability
maps into one ensemble prediction per test image, and reports segmentation metrics
against ground truth polygons.
"""

import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from finetuning import (
    build_uke_pool_and_test_holdout,
    compute_batch_metrics,
    load_model,
    load_training_config,
)

GROUP_KEYS = [
    "scenario_name",
    "lr",
    "aug_per_img",
    "seed",
    "train_domain_tag",
    "val_domain_tag",
]


def load_and_preprocess(image_path):
    """Load an image and resize it to the 320x320 input the model expects."""
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = np.array(pil_img, dtype=np.float32) / 255.0
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
    tensor = (
        torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    )
    return tensor, h, w


def predict_probability_map(net, tensor, device, h, w):
    """Run one model and return its probability map resampled back to the original size."""
    with torch.no_grad():
        d0, _, _, _, _, _, _ = net(tensor.to(device))
        prob_320 = d0[0, 0].detach().cpu().numpy()

    prob_orig = cv2.resize(prob_320, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(prob_orig, 0, 1).astype(np.float32)


def predict_ensemble_probability_map(nets, tensor, device, h, w):
    """Average probability maps from multiple models (ensemble across folds)."""
    probs = [predict_probability_map(net, tensor, device, h, w) for net in nets]
    return np.mean(np.stack(probs, axis=0), axis=0)


def polygon_to_mask(polygon, h, w):
    """Rasterize a polygon into a binary mask matching image height/width."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return mask


def append_row(row, csv_path):
    """Append one result row to a CSV, creating it if needed."""
    df_row = pd.DataFrame([row])
    csv_file = Path(csv_path)
    if not csv_file.exists():
        df_row.to_csv(csv_file, index=False)
    else:
        df_row.to_csv(csv_file, mode="a", header=False, index=False)


def evaluate_group(group_label, nets, test_paths, polygon_test, device, per_image_csv):
    """Evaluate one model (or ensemble of fold-models) over the held-out test set."""
    rows = []
    for idx, image_path in enumerate(test_paths):
        tensor, h, w = load_and_preprocess(image_path)
        t0 = time.time()
        avg_prob = predict_ensemble_probability_map(nets, tensor, device, h, w)
        dt = time.time() - t0

        pred_mask = (avg_prob > 0.5).astype(np.uint8)
        gt_mask = polygon_to_mask(polygon_test[image_path], h, w)

        pred_t = (
            torch.from_numpy(pred_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )
        gt_t = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        metrics = compute_batch_metrics(pred_t, gt_t, thresh=0.5)
        selection_score = 0.7 * metrics["f2"] + 0.3 * metrics["dice"]

        row = {
            "group_label": group_label,
            "image_path": image_path,
            "n_models": len(nets),
            "inference_sec": dt,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "f2": metrics["f2"],
            "selection_score": selection_score,
        }
        rows.append(row)
        append_row(row, per_image_csv)

        if (idx + 1) % 25 == 0:
            print(f"  [{group_label}] {idx + 1}/{len(test_paths)}")

    return rows


def main():
    training_config = load_training_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = training_config.get(
        "checkpoint_dir",
        "/workspaces/de.uke.iam.soos-document-ai-lab/train_uke_model/uke_models",
    )
    manifest_path = training_config.get(
        "manifest_path", os.path.join(checkpoint_dir, "training_runs_manifest.csv")
    )
    eval_output_dir = training_config.get(
        "eval_output_dir", os.path.join(checkpoint_dir, "test_eval")
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    per_image_csv = os.path.join(eval_output_dir, "test_ensemble_per_image.csv")
    summary_csv = os.path.join(eval_output_dir, "test_ensemble_summary.csv")

    if os.path.exists(per_image_csv):
        os.remove(per_image_csv)
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    print("Reconstructing held-out UKE test set...")
    uke_holdout = build_uke_pool_and_test_holdout(training_config)
    test_paths = uke_holdout["test_paths"]
    polygon_test = uke_holdout["polygon_test"]
    print(f"Held-out UKE test images: {len(test_paths)}")
    if len(test_paths) == 0:
        raise ValueError("No held-out UKE test images with ground truth found.")

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Checkpoint manifest not found at {manifest_path}. Run finetuning.py first."
        )
    manifest_df = pd.read_csv(manifest_path)

    summary_rows = []

    # --- Baseline: pretrained model, no fine-tuning ---
    model_path = training_config.get(
        "model_path", "/workspaces/de.uke.iam.soos-document-ai-lab/u2net.pth"
    )
    print(f"\n=== Evaluating baseline (pretrained, {model_path}) ===")
    net = load_model(model_path, device)
    net.eval()
    try:
        rows = evaluate_group(
            "pretrained", [net], test_paths, polygon_test, device, per_image_csv
        )
    finally:
        del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_rows = pd.DataFrame(rows)
    summary_rows.append(
        {
            "group_label": "pretrained",
            "scenario_name": None,
            "lr": None,
            "n_models": 1,
            "n_images": len(df_rows),
            **df_rows[["precision", "recall", "iou", "dice", "f2", "selection_score"]]
            .mean()
            .to_dict(),
        }
    )

    # --- Fine-tuned fold ensembles, grouped by scenario/lr/aug/seed/domain ---
    for group_values, group_df in manifest_df.groupby(GROUP_KEYS, dropna=False):
        group_dict = dict(zip(GROUP_KEYS, group_values))
        checkpoint_paths = sorted(group_df["checkpoint_path"].unique().tolist())
        group_label = (
            f"finetuned__{group_dict['scenario_name']}"
            f"__lr-{group_dict['lr']}"
            f"__aug-{group_dict['aug_per_img']}"
        )
        print(
            f"\n=== Evaluating ensemble '{group_label}' "
            f"({len(checkpoint_paths)} fold model(s)) ==="
        )

        nets = []
        try:
            for checkpoint_path in checkpoint_paths:
                if not os.path.isfile(checkpoint_path):
                    print(
                        f"  WARN: checkpoint missing on disk, skipping: {checkpoint_path}"
                    )
                    continue
                net = load_model(checkpoint_path, device)
                net.eval()
                nets.append(net)

            if len(nets) == 0:
                print("  WARN: no checkpoints found for this group, skipping.")
                continue

            rows = evaluate_group(
                group_label, nets, test_paths, polygon_test, device, per_image_csv
            )
        finally:
            for net in nets:
                del net
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df_rows = pd.DataFrame(rows)
        summary_rows.append(
            {
                "group_label": group_label,
                **group_dict,
                "n_models": len(nets),
                "n_images": len(df_rows),
                **df_rows[
                    ["precision", "recall", "iou", "dice", "f2", "selection_score"]
                ]
                .mean()
                .to_dict(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary written to {summary_csv}")
    print(f"Per-image results written to {per_image_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()
