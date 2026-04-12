"""Sync images to/from Hugging Face dataset repo."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .storage import shard_for

REPO_ID = "wt200999/apriltag-dataset"
REPO_TYPE = "dataset"


def _build_staging(data_dir: Path) -> Path:
    """Build a staging directory that mirrors the HF repo layout.

    Creates train/<shard>/ symlinks pointing to data/images/<shard>/,
    plus a train/metadata.jsonl with shard-prefixed file_name values.
    """
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"
    staging = data_dir / ".hf-staging"
    train_dir = staging / "train"

    if train_dir.exists():
        shutil.rmtree(str(train_dir))
    train_dir.mkdir(parents=True)

    for child in sorted(images_dir.iterdir()):
        if child.is_dir() and len(child.name) == 2:
            (train_dir / child.name).symlink_to(child.resolve())

    count = 0
    with open(train_dir / "metadata.jsonl", "w") as out:
        for det_file in sorted(detections_dir.iterdir()):
            if det_file.suffix != ".json":
                continue
            with open(det_file) as f:
                det = json.load(f)
            image_file = det["image_file"]
            shard = shard_for(image_file)
            if not (images_dir / shard / image_file).exists():
                continue
            row = {
                "file_name": f"{shard}/{image_file}",
                "image_sha256": det["image_sha256"],
                "image_width": det["image_width"],
                "image_height": det["image_height"],
                "num_detections": det["num_detections"],
                "detections": det["detections"],
            }
            out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} entries to staging metadata.jsonl")
    return staging


def upload_images(data_dir: Path) -> None:
    """Upload images + metadata to Hugging Face under train/ prefix."""
    from huggingface_hub import HfApi

    images_dir = data_dir / "images"
    if not images_dir.exists():
        print("No images directory found.")
        return

    image_files = list(images_dir.rglob("*.png"))
    if not image_files:
        print("No images to upload.")
        return

    staging = _build_staging(data_dir)
    print(f"Uploading {len(image_files)} images to {REPO_ID}...")

    api = HfApi()
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(staging),
        allow_patterns=["train/**/*.png", "train/metadata.jsonl"],
    )
    print("Upload complete.")


def download_images(data_dir: Path) -> None:
    """Download images from Hugging Face."""
    from huggingface_hub import snapshot_download

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.name for f in images_dir.rglob("*.png"))
    print(f"Downloading images from {REPO_ID}...")
    print(f"  ({len(existing)} images already present locally)")

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=["train/**/*.png", "train/*/*.png"],
            local_dir=tmp,
        )

        train_dir = Path(tmp) / "train"
        if train_dir.exists():
            new = 0
            for f in train_dir.rglob("*.png"):
                shard_dir = images_dir / shard_for(f.name)
                shard_dir.mkdir(exist_ok=True)
                dest = shard_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
                    new += 1
            print(f"Download complete. {new} new image(s).")
        else:
            print("No images found on remote.")
