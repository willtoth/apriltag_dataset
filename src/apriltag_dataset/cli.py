from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .detect import create_detector, detect_image
from .ingest import ingest_source
from .storage import (
    image_path,
    load_manifest,
    read_detection,
    rebuild_manifest,
    save_manifest,
    write_detection,
)


def cmd_ingest(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    detector = create_detector(
        family=args.family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
    )
    manifest = load_manifest(data_dir)

    total = 0
    for p in args.paths:
        count = ingest_source(
            p, data_dir, detector, args.family, manifest,
            skip_empty=not args.keep_empty,
            frame_interval=args.frame_interval,
            face_action=args.face_action,
        )
        total += count

    save_manifest(data_dir, manifest)
    print(f"\nIngested {total} new image(s). Dataset now has {len(manifest)} image(s).")


def cmd_regenerate(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"

    if not images_dir.exists():
        print("No images directory found.")
        return

    image_files = sorted(images_dir.rglob("*.png"))
    if not image_files:
        print("No images found.")
        return

    detector = create_detector(
        family=args.family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
    )

    from PIL import Image
    import numpy as np
    from .ingest import compute_sha256

    changed = 0
    for img_path in image_files:
        img = Image.open(img_path).convert("L")
        sha256 = compute_sha256(img_path.read_bytes())

        old_det_path = detections_dir / f"{img_path.stem}.json"
        old_num = 0
        if old_det_path.exists():
            old_result = read_detection(old_det_path)
            old_num = old_result.num_detections

        result = detect_image(detector, img, img_path.name, sha256, args.family)
        write_detection(result, detections_dir)

        if result.num_detections != old_num:
            changed += 1
            print(
                f"  CHANGED {img_path.name}: {old_num} -> {result.num_detections} detection(s)"
            )
        else:
            print(f"  OK      {img_path.name}: {result.num_detections} detection(s)")

    print(f"\nRegenerated {len(image_files)} image(s). {changed} changed.")

    # Rebuild manifest to update detection counts
    rebuild_manifest(data_dir)


def cmd_manifest(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    entries = rebuild_manifest(data_dir)
    print(f"Manifest rebuilt with {len(entries)} image(s).")


def cmd_stats(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    manifest = load_manifest(data_dir)

    if not manifest:
        print("Dataset is empty.")
        return

    total_detections = sum(e.num_detections for e in manifest)
    sources = Counter(e.source for e in manifest)
    with_detections = sum(1 for e in manifest if e.num_detections > 0)
    without_detections = sum(1 for e in manifest if e.num_detections == 0)

    # Gather tag family info from detection files
    detections_dir = data_dir / "detections"
    families: Counter[str] = Counter()
    tag_ids: Counter[int] = Counter()
    if detections_dir.exists():
        for det_file in detections_dir.iterdir():
            if det_file.suffix == ".json":
                result = read_detection(det_file)
                for det in result.detections:
                    families[det.tag_family] += 1
                    tag_ids[det.tag_id] += 1

    print(f"Images:           {len(manifest)}")
    print(f"  with tags:      {with_detections}")
    print(f"  without tags:   {without_detections}")
    print(f"Total detections: {total_detections}")

    if families:
        print(f"\nTag families:")
        for fam, count in families.most_common():
            print(f"  {fam}: {count}")

    if tag_ids:
        print(f"\nUnique tag IDs:   {len(tag_ids)}")

    if sources:
        print(f"\nSources:")
        for src, count in sources.most_common():
            print(f"  {src}: {count}")


def cmd_prune(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"
    manifest = load_manifest(data_dir)

    if not manifest:
        print("Dataset is empty.")
        return

    empty = [e for e in manifest if e.num_detections == 0]
    if not empty:
        print("No images with 0 detections found.")
        return

    print(f"Found {len(empty)} image(s) with 0 detections.")
    removed = 0
    for entry in empty:
        img_path = image_path(images_dir, entry.filename)
        det_path = detections_dir / f"{Path(entry.filename).stem}.json"

        if img_path.exists():
            img_path.unlink()
        if det_path.exists():
            det_path.unlink()

        removed += 1
        print(f"  REMOVE {entry.filename}")

    kept = [e for e in manifest if e.num_detections > 0]
    save_manifest(data_dir, kept)
    print(f"\nPruned {removed} image(s). Dataset now has {len(kept)} image(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="apriltag-dataset",
        description="AprilTag image dataset tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest images into the dataset")
    p_ingest.add_argument("paths", nargs="+", help="Image files, archives, directories, or URLs")
    p_ingest.add_argument("--family", default="tag36h11", help="Tag family (default: tag36h11)")
    p_ingest.add_argument("--nthreads", type=int, default=4, help="Detector threads (default: 4)")
    p_ingest.add_argument("--quad-decimate", type=float, default=1.0, help="Quad decimate (default: 1.0, no decimation)")
    p_ingest.add_argument("--keep-empty", action="store_true", help="Keep images with 0 detections (default: skip them)")
    p_ingest.add_argument("--frame-interval", type=float, default=2.0, help="Seconds between extracted video frames (default: 2.0)")
    p_ingest.add_argument("--face-action", choices=["blur", "skip", "none"], default="blur", help="Face handling for video frames: blur (default), skip, or none")
    p_ingest.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_ingest.set_defaults(func=cmd_ingest)

    # regenerate
    p_regen = sub.add_parser("regenerate", help="Re-run detections on all images")
    p_regen.add_argument("--family", default="tag36h11", help="Tag family (default: tag36h11)")
    p_regen.add_argument("--nthreads", type=int, default=4, help="Detector threads (default: 4)")
    p_regen.add_argument("--quad-decimate", type=float, default=1.0, help="Quad decimate (default: 1.0, no decimation)")
    p_regen.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_regen.set_defaults(func=cmd_regen)

    # manifest
    p_manifest = sub.add_parser("manifest", help="Rebuild manifest.json")
    p_manifest.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_manifest.set_defaults(func=cmd_manifest)

    # prune
    p_prune = sub.add_parser("prune", help="Remove images with 0 detections")
    p_prune.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_prune.set_defaults(func=cmd_prune)

    # review
    p_review = sub.add_parser("review", help="Launch dataset review website")
    p_review.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_review.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    p_review.set_defaults(func=cmd_review)

    # stats
    p_stats = sub.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_stats.set_defaults(func=cmd_stats)

    # upload
    p_upload = sub.add_parser("upload", help="Upload images to Hugging Face")
    p_upload.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_upload.set_defaults(func=cmd_upload)

    # download
    p_download = sub.add_parser("download", help="Download images from Hugging Face")
    p_download.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_download.set_defaults(func=cmd_download)

    # repair
    p_repair = sub.add_parser("repair", help="Backfill ingest metadata into detection sidecars and rebuild manifest")
    p_repair.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_repair.set_defaults(func=cmd_repair)

    # migrate
    p_migrate = sub.add_parser("migrate", help="Move flat images into sharded subdirectories")
    p_migrate.add_argument("--data-dir", default="./data", help="Data directory (default: ./data)")
    p_migrate.set_defaults(func=cmd_migrate)

    args = parser.parse_args()
    args.func(args)


def cmd_review(args: argparse.Namespace) -> None:
    from .server import run_server
    run_server(Path(args.data_dir), port=args.port)


def cmd_upload(args: argparse.Namespace) -> None:
    from .sync import upload_images
    data_dir = Path(args.data_dir).resolve()
    upload_images(data_dir)


def cmd_download(args: argparse.Namespace) -> None:
    from .sync import download_images
    download_images(Path(args.data_dir).resolve())


def cmd_repair(args: argparse.Namespace) -> None:
    from datetime import datetime, timezone

    import imagehash
    from PIL import Image

    from .schema import IngestMetadata
    from .storage import read_detection, rebuild_manifest, write_detection

    data_dir = Path(args.data_dir).resolve()
    images_dir = data_dir / "images"
    detections_dir = data_dir / "detections"

    count = 0
    skipped = 0
    for img_path in sorted(images_dir.rglob("*.png")):
        det_path = detections_dir / f"{img_path.stem}.json"
        if not det_path.exists():
            continue

        result = read_detection(det_path)
        if result.ingest_metadata is not None:
            skipped += 1
            continue

        img = Image.open(img_path)
        dhash_hex = str(imagehash.dhash(img))

        parts = img_path.stem.split("_", 1)
        original_name = (parts[1] + ".png") if len(parts) > 1 else img_path.name

        mtime = img_path.stat().st_mtime
        ingested_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        result.ingest_metadata = IngestMetadata(
            original_name=original_name,
            source="unknown",
            dhash=dhash_hex,
            ingested_at=ingested_at,
        )
        write_detection(result, detections_dir)
        count += 1
        if count % 500 == 0:
            print(f"  Repaired {count} sidecars...")

    print(f"\nRepaired {count} sidecar(s), skipped {skipped} (already had metadata).")

    entries = rebuild_manifest(data_dir)
    print(f"Manifest rebuilt with {len(entries)} entries.")


def cmd_migrate(args: argparse.Namespace) -> None:
    from .storage import shard_for

    data_dir = Path(args.data_dir).resolve()
    images_dir = data_dir / "images"

    if not images_dir.exists():
        print("No images directory found.")
        return

    flat_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
    if not flat_files:
        print("No flat images to migrate.")
        return

    print(f"Migrating {len(flat_files)} images into shard subdirectories...")
    for f in flat_files:
        shard_dir = images_dir / shard_for(f.name)
        shard_dir.mkdir(exist_ok=True)
        f.rename(shard_dir / f.name)

    print(f"Migrated {len(flat_files)} image(s).")


def cmd_regen(args: argparse.Namespace) -> None:
    cmd_regenerate(args)
