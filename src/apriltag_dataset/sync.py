"""Sync images to/from Hugging Face dataset repo."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

REPO_ID = "wt200999/apriltag-dataset"
REPO_TYPE = "dataset"


def upload_images(data_dir: Path) -> None:
    """Upload images + metadata to Hugging Face under train/ prefix."""
    from huggingface_hub import HfApi

    images_dir = data_dir / "images"
    if not images_dir.exists():
        print("No images directory found.")
        return

    image_files = list(images_dir.glob("*.png"))
    if not image_files:
        print("No images to upload.")
        return

    print(f"Uploading {len(image_files)} images to {REPO_ID}...")
    api = HfApi()
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(images_dir),
        path_in_repo="train",
        allow_patterns=["*.png", "metadata.jsonl"],
    )
    print("Upload complete.")


def download_images(data_dir: Path) -> None:
    """Download images from Hugging Face."""
    from huggingface_hub import snapshot_download

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.name for f in images_dir.glob("*.png"))
    print(f"Downloading images from {REPO_ID}...")
    print(f"  ({len(existing)} images already present locally)")

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=["train/*.png"],
            local_dir=tmp,
        )

        train_dir = Path(tmp) / "train"
        if train_dir.exists():
            new = 0
            for f in train_dir.glob("*.png"):
                dest = images_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    new += 1
            print(f"Download complete. {new} new image(s).")
        else:
            print("No images found on remote.")
