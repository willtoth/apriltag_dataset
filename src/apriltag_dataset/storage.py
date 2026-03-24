from __future__ import annotations

import json
from pathlib import Path

from .schema import ImageDetectionResult, ManifestEntry


def write_detection(result: ImageDetectionResult, detections_dir: Path) -> Path:
    detections_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_file).stem
    out_path = detections_dir / f"{stem}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    return out_path


def read_detection(json_path: Path) -> ImageDetectionResult:
    data = json.loads(json_path.read_text())
    return ImageDetectionResult.from_dict(data)


def load_manifest(data_dir: Path) -> list[ManifestEntry]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text())
    return [ManifestEntry.from_dict(entry) for entry in data.get("images", [])]


def save_manifest(data_dir: Path, entries: list[ManifestEntry]) -> None:
    manifest_path = data_dir / "manifest.json"
    data = {
        "dataset_version": "1.0",
        "total_images": len(entries),
        "images": [e.to_dict() for e in entries],
    }
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")


def rebuild_manifest(data_dir: Path) -> list[ManifestEntry]:
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"
    entries: list[ManifestEntry] = []

    if not images_dir.exists():
        save_manifest(data_dir, entries)
        return entries

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        det_path = detections_dir / f"{img_path.stem}.json"
        num_dets = 0
        sha256 = ""
        dhash_hex = ""
        width = 0
        height = 0
        if det_path.exists():
            result = read_detection(det_path)
            num_dets = result.num_detections
            sha256 = result.image_sha256
            width = result.image_width
            height = result.image_height

        # Reconstruct original name from the hash-prefixed filename
        # Format: {sha256_12}_{original}.png
        parts = img_path.stem.split("_", 1)
        original_name = parts[1] if len(parts) > 1 else img_path.stem

        entries.append(
            ManifestEntry(
                filename=img_path.name,
                original_name=original_name,
                source="unknown",
                sha256=sha256,
                dhash=dhash_hex,
                width=width,
                height=height,
                ingested_at="",
                num_detections=num_dets,
            )
        )

    save_manifest(data_dir, entries)
    return entries
