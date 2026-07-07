"""Violin plot of per-image selection_score, comparing pretrained vs fine-tuned ensembles.

Reads the per-image CSV written by evaluate_ensemble.py (one row per test image per
group_label) and plots the selection_score distribution per group as a violin plot.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC = "selection_score"


def clean_label(group_label):
    """Turn a raw group_label into a short, readable legend/tick label."""
    if group_label == "pretrained":
        return "Pretrained U$^2$-Net"
    if group_label.startswith("finetuned__"):
        parts = group_label.split("__")
        scenario = parts[1] if len(parts) > 1 else group_label
        return f"Fine-Tuned U$^2$-Net\n({scenario})"
    return group_label


def plot_violin(df, output_dir, metric=METRIC):
    """Draw and save the violin plot for one metric, grouped by group_label."""
    order = sorted(df["group_label"].unique(), key=lambda g: g != "pretrained")
    labels = [clean_label(g) for g in order]
    data = [df[df["group_label"] == g][metric].dropna().values for g in order]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(order))]

    fig, ax = plt.subplots(figsize=(1.8 * len(order) + 2, 4.5))

    vp = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)

    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)

    vp["cmedians"].set_colors(colors)
    vp["cbars"].set_colors(colors)
    vp["cmins"].set_colors(colors)
    vp["cmaxes"].set_colors(colors)

    for i, vals in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, s=14, alpha=0.5, color=colors[i], edgecolors="black", linewidths=0.3)

    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"violin_{metric}.png")
    pdf_path = os.path.join(output_dir, f"violin_{metric}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-image-csv",
        default=None,
        help="Path to test_ensemble_per_image.csv from evaluate_ensemble.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the violin plot into (defaults to the CSV's directory)",
    )
    args = parser.parse_args()

    per_image_csv = args.per_image_csv
    if per_image_csv is None:
        from finetuning import load_training_config

        training_config = load_training_config()
        checkpoint_dir = training_config.get("checkpoint_dir", ".")
        eval_output_dir = training_config.get(
            "eval_output_dir", os.path.join(checkpoint_dir, "test_eval")
        )
        per_image_csv = os.path.join(eval_output_dir, "test_ensemble_per_image.csv")

    output_dir = args.output_dir or os.path.dirname(per_image_csv)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(per_image_csv)
    plot_violin(df, output_dir)


if __name__ == "__main__":
    main()
