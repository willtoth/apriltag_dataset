"""Generate metadata.jsonl for Hugging Face ImageFolder dataset format."""

from __future__ import annotations

import json
from pathlib import Path


def generate_metadata(data_dir: Path) -> int:
    """Generate metadata.jsonl pairing images with detection data.

    Creates data/metadata.jsonl for HF ImageFolder auto-loading.
    """
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"
    metadata_path = images_dir / "metadata.jsonl"

    if not images_dir.exists() or not detections_dir.exists():
        print("Missing images/ or detections/ directory.")
        return 0

    count = 0
    with open(metadata_path, "w") as out:
        for det_file in sorted(detections_dir.iterdir()):
            if det_file.suffix != ".json":
                continue

            with open(det_file) as f:
                det = json.load(f)

            image_file = det["image_file"]
            if not (images_dir / image_file).exists():
                continue

            row = {
                "file_name": image_file,
                "image_sha256": det["image_sha256"],
                "image_width": det["image_width"],
                "image_height": det["image_height"],
                "num_detections": det["num_detections"],
                "detections": det["detections"],
            }
            out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} entries to {metadata_path}")
    return count
