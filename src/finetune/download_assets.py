"""Config-driven setup: fetch the pretrained U2NET weights and the SmartDoc15 dataset.

Reads paths from training_config.json (see load_training_config) so this script and
finetuning.py always agree on where things live. Run it once before finetuning.py.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

from download_smartdoc15_dataset import download_and_extract
from finetuning import load_training_config

DEFAULT_U2NET_URL = (
    "https://huggingface.co/lilpotat/pytorch3d/resolve/"
    "346374a95673795896e94398d65700cb19199e31/u2net.pth"
)


def download_u2net_weights(model_path: Path, url: str, force: bool = False) -> None:
    """Download the pretrained U2NET checkpoint to model_path if it is not already there."""
    if model_path.is_file() and not force:
        print(f"u2net.pth already present at {model_path}, skipping (use --force to redownload).")
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading u2net.pth from {url}")
    request = Request(url, headers={"User-Agent": "semedo"})
    with urlopen(request, timeout=120) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(model_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = 100 * downloaded / total_size
                    print(f"  {downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="\r")
    print(f"\nSaved u2net.pth to {model_path}")

    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or len(checkpoint) == 0:
        raise RuntimeError(f"Downloaded file at {model_path} is not a valid U2NET state dict.")
    print(f"Verified checkpoint: {len(checkpoint)} tensors.")


def download_smartdoc15(root_folder: Path, metadata_path: Path, force: bool = False) -> None:
    """Download and extract the SmartDoc15 frames archive if it is not already there."""
    if metadata_path.is_file() and not force:
        print(f"SmartDoc15 metadata already present at {metadata_path}, skipping (use --force to redownload).")
        return

    # The release archive ships metadata.csv.gz, not a plain frames_metadata.csv,
    # so decompress it once extraction has produced it.
    gz_path = root_folder / "metadata.csv.gz"
    if force or not gz_path.is_file():
        download_and_extract("frames.tar.gz", root_folder)

    if not metadata_path.is_file() or force:
        if not gz_path.is_file():
            raise RuntimeError(
                f"Extraction finished but metadata archive is still missing: {gz_path}"
            )
        print(f"Decompressing {gz_path} to {metadata_path}")
        with gzip.open(gz_path, "rb") as src, open(metadata_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the pretrained U2NET weights and the SmartDoc15 dataset "
        "into the paths configured in training_config.json."
    )
    parser.add_argument("--config", default=None, help="Path to training_config.json (defaults to the one next to this script).")
    parser.add_argument("--u2net-url", default=None, help="Override the u2net.pth download URL (defaults to training_config.json's u2net_download_url, or the built-in HF mirror).")
    parser.add_argument("--skip-model", action="store_true", help="Do not download u2net.pth.")
    parser.add_argument("--skip-data", action="store_true", help="Do not download the SmartDoc15 dataset.")
    parser.add_argument("--force", action="store_true", help="Redownload even if the target already exists.")
    args = parser.parse_args()

    config = load_training_config(args.config)

    if not args.skip_model:
        model_path = Path(config.get("model_path", "../../models/u2net.pth")).expanduser()
        url = args.u2net_url or config.get("u2net_download_url") or DEFAULT_U2NET_URL
        download_u2net_weights(model_path, url, force=args.force)

    if not args.skip_data:
        root_folder = Path(config.get("root_folder", "../../data/external/smartdoc15")).expanduser()
        metadata_path = Path(
            config.get("opensource_metadata_path", root_folder / "frames_metadata.csv")
        ).expanduser()
        download_smartdoc15(root_folder, metadata_path, force=args.force)

    print("\n===== SETUP DONE =====")


if __name__ == "__main__":
    main()
