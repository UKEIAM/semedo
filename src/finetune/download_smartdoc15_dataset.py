"""Download SmartDoc15 CH1 release assets locally.

The dataset stays outside the repository and is only referenced via config.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

LATEST_RELEASE_API = (
    "https://api.github.com/repos/jchazalon/smartdoc15-ch1-dataset/releases/latest"
)
DEFAULT_ASSET = "frames.tar.gz"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "data" / "external" / "smartdoc15"

# Name the release archive ships the metadata under, and the name training_config.json
# (opensource_metadata_path) expects it as.
BUNDLED_METADATA_NAME = "metadata.csv.gz"
EXPECTED_METADATA_NAME = "frames_metadata.csv"


def fetch_release_asset_url(asset_name: str) -> str:
    request = Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "semedo"},
    )
    with urlopen(request, timeout=60) as response:
        release = json.loads(response.read().decode("utf-8"))

    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            return asset["browser_download_url"]

    raise RuntimeError(f"Asset '{asset_name}' not found in the latest release.")


def extract_safe(archive_path: Path, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_dir)):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(target_dir)


def decompress_bundled_metadata(target_dir: Path) -> None:
    """Decompress metadata.csv.gz (as shipped in the archive) into frames_metadata.csv
    (the name training_config.json's opensource_metadata_path expects).
    """
    gz_path = target_dir / BUNDLED_METADATA_NAME
    csv_path = target_dir / EXPECTED_METADATA_NAME
    if not gz_path.is_file():
        return
    print(f"Decompressing {gz_path.name} -> {csv_path.name}")
    with gzip.open(gz_path, "rb") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst)


def download_and_extract(asset_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    asset_url = fetch_release_asset_url(asset_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / asset_name
        print(f"Downloading {asset_name} from {asset_url}")
        urlretrieve(asset_url, archive_path)
        print(f"Extracting to {target_dir}")
        extract_safe(archive_path, target_dir)

    decompress_bundled_metadata(target_dir)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SmartDoc15 CH1 release assets."
    )
    parser.add_argument(
        "--asset",
        default=DEFAULT_ASSET,
        help="Release asset to download (frames.tar.gz or models.tar.gz).",
    )
    parser.add_argument(
        "--target-dir",
        default=str(DEFAULT_TARGET_DIR),
        help=(
            "Directory where the archive should be extracted "
            f"(default: {DEFAULT_TARGET_DIR}, resolved relative to this script, "
            "not your current working directory)."
        ),
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir).expanduser().resolve()
    download_and_extract(args.asset, target_dir)
    print(f"Done: {target_dir}")


if __name__ == "__main__":
    main()
