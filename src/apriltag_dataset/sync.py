"""Sync images to/from Hugging Face dataset repo."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ID = "wt200999/apriltag-dataset"
REPO_TYPE = "dataset"


def upload_images(data_dir: Path) -> None:
    """Upload images directory to Hugging Face."""
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
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(data_dir),
        allow_patterns=["images/*.png"],
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

    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=["images/*.png"],
        local_dir=str(data_dir),
    )

    new_count = len(list(images_dir.glob("*.png"))) - len(existing)
    print(f"Download complete. {new_count} new image(s).")
