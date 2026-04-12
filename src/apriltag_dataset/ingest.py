from __future__ import annotations

import hashlib
import io
import re
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image
from pupil_apriltags import Detector

from .detect import detect_image
from .schema import ManifestEntry
from .storage import load_manifest, read_detection, save_manifest, shard_for, write_detection
from .video import (
    VIDEO_EXTENSIONS,
    download_video,
    extract_frames,
    is_playlist_url,
    is_video_file,
    is_video_url,
    iter_playlist_entries,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
DHASH_THRESHOLD = 5
AREA_DIFF_THRESHOLD = 0.25


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def sanitize_name(name: str, max_len: int = 200) -> str:
    stem = Path(name).stem
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return sanitized[:max_len]


def preprocess_image(path: Path) -> tuple[Image.Image, bytes]:
    img = Image.open(path)
    img = img.convert("L")
    img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return img, buf.getvalue()


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_dhash(img: Image.Image) -> str:
    return str(imagehash.dhash(img))


def find_near_duplicate(
    dhash_hex: str, width: int, height: int, manifest: list[ManifestEntry]
) -> ManifestEntry | None:
    new_hash = imagehash.hex_to_hash(dhash_hex)
    new_area = width * height

    for entry in manifest:
        if not entry.dhash:
            continue
        existing_hash = imagehash.hex_to_hash(entry.dhash)
        distance = new_hash - existing_hash
        if distance <= DHASH_THRESHOLD:
            existing_area = entry.width * entry.height
            if existing_area > 0:
                area_ratio = abs(new_area - existing_area) / max(new_area, existing_area)
                if area_ratio <= AREA_DIFF_THRESHOLD:
                    return entry
    return None


def process_single_image(
    image_path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    source_name: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
) -> ManifestEntry | None:
    if not is_image_file(image_path):
        return None

    try:
        grey_img, png_bytes = preprocess_image(image_path)
    except Exception as e:
        print(f"  SKIP {image_path.name}: failed to open ({e})")
        return None

    sha256 = compute_sha256(png_bytes)
    sha_prefix = sha256[:12]

    images_dir = data_dir / "images"
    dhash_hex = compute_dhash(grey_img)
    w, h = grey_img.size
    sanitized = sanitize_name(image_path.name)
    dest_name = f"{sha_prefix}_{sanitized}.png"

    # Check for exact duplicate — allow re-ingest if previous had 0 detections
    shard_dir = images_dir / shard_for(dest_name)
    existing = list(shard_dir.glob(f"{sha_prefix}_*")) if shard_dir.exists() else []
    if existing:
        existing_stem = existing[0].stem
        det_path = data_dir / "detections" / f"{existing_stem}.json"
        if det_path.exists():
            old_result = read_detection(det_path)
            if old_result.num_detections > 0:
                print(f"  SKIP {image_path.name}: exact duplicate of {existing[0].name}")
                return None
            print(f"  RETRY {image_path.name}: re-running detection on {existing[0].name}")
            dest_name = existing[0].name
        else:
            print(f"  RETRY {image_path.name}: missing detection for {existing[0].name}")
            dest_name = existing[0].name
    else:
        # Near-duplicate check (only for new images)
        dup = find_near_duplicate(dhash_hex, w, h, manifest)
        if dup is not None:
            # Allow re-ingest if previous had 0 detections
            det_path = data_dir / "detections" / f"{Path(dup.filename).stem}.json"
            if det_path.exists():
                old_result = read_detection(det_path)
                if old_result.num_detections > 0:
                    print(f"  SKIP {image_path.name}: near-duplicate of {dup.filename}")
                    return None

    # Run detection before writing anything to disk
    try:
        result = detect_image(detector, grey_img, dest_name, sha256, family)
    except Exception as e:
        print(f"  SKIP {image_path.name}: detection failed ({e})")
        return None

    if skip_empty and result.num_detections == 0:
        print(f"  SKIP {image_path.name}: no tags detected (--skip-empty)")
        return None

    from .schema import IngestMetadata
    result.ingest_metadata = IngestMetadata(
        original_name=image_path.name,
        source=source_name,
        dhash=dhash_hex,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )

    # Write image and sidecar
    shard_dir.mkdir(parents=True, exist_ok=True)
    dest_path = shard_dir / dest_name
    dest_path.write_bytes(png_bytes)

    detections_dir = data_dir / "detections"
    write_detection(result, detections_dir)

    action = "ADD" if not existing else "UPDATE"
    print(
        f"  {action:6s} {image_path.name} -> {dest_name} "
        f"({result.num_detections} detection(s))"
    )

    # Remove old manifest entry if re-ingesting
    if existing:
        manifest[:] = [e for e in manifest if e.filename != dest_name]

    entry = ManifestEntry(
        filename=dest_name,
        original_name=image_path.name,
        source=source_name,
        sha256=sha256,
        dhash=dhash_hex,
        width=w,
        height=h,
        ingested_at=datetime.now(timezone.utc).isoformat(),
        num_detections=result.num_detections,
    )
    manifest.append(entry)
    return entry


def ingest_video(
    video_path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
    frame_interval: float = 2.0,
    face_action: str = "blur",
) -> int:
    source_name = video_path.name
    count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir)
        for frame_path, label in extract_frames(
            video_path, frames_dir,
            interval_seconds=frame_interval,
            face_action=face_action,
            dhash_threshold=DHASH_THRESHOLD,
        ):
            result = process_single_image(
                frame_path, data_dir, detector, family, source_name, manifest,
                skip_empty=skip_empty,
            )
            if result is not None:
                count += 1
            # Clean up temp frame immediately
            frame_path.unlink(missing_ok=True)
    return count


def ingest_directory(
    dir_path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    source_name: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
) -> int:
    count = 0
    for child in sorted(dir_path.rglob("*")):
        if child.is_file() and is_image_file(child):
            result = process_single_image(
                child, data_dir, detector, family, source_name, manifest,
                skip_empty=skip_empty,
            )
            if result is not None:
                count += 1
    return count


def ingest_zip(
    zip_path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        return ingest_directory(
            Path(tmpdir), data_dir, detector, family, zip_path.name, manifest,
            skip_empty=skip_empty,
        )


def ingest_tar(
    tar_path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
) -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmpdir, filter="data")
        return ingest_directory(
            Path(tmpdir), data_dir, detector, family, tar_path.name, manifest,
            skip_empty=skip_empty,
        )


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path)
    name = Path(path).name
    return name if name else "download"


def download_to_tempdir(url: str, tmpdir: Path) -> Path:
    filename = _filename_from_url(url)
    dest = tmpdir / filename
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def ingest_source(
    source: str,
    data_dir: Path,
    detector: Detector,
    family: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
    frame_interval: float = 2.0,
    face_action: str = "blur",
) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "detections").mkdir(exist_ok=True)

    if _is_url(source):
        if is_playlist_url(source):
            total_count = 0
            for i, total, video_url, title in iter_playlist_entries(source):
                print(f"\n[{i + 1}/{total}] {title}")
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        video_path = download_video(video_url, Path(tmpdir))
                        count = ingest_video(
                            video_path, data_dir, detector, family, manifest,
                            skip_empty=skip_empty,
                            frame_interval=frame_interval,
                            face_action=face_action,
                        )
                        total_count += count
                except Exception as e:
                    print(f"  ERROR: Failed to process video: {e}")
                    continue
            return total_count

        if is_video_url(source):
            with tempfile.TemporaryDirectory() as tmpdir:
                print(f"Ingesting video URL: {source}")
                video_path = download_video(source, Path(tmpdir))
                return ingest_video(
                    video_path, data_dir, detector, family, manifest,
                    skip_empty=skip_empty,
                    frame_interval=frame_interval,
                    face_action=face_action,
                )
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = download_to_tempdir(source, Path(tmpdir))
            return _ingest_local(downloaded, data_dir, detector, family, manifest,
                                 skip_empty=skip_empty,
                                 frame_interval=frame_interval,
                                 face_action=face_action)

    path = Path(source).resolve()
    if not path.exists():
        print(f"Path not found: {source}")
        return 0
    return _ingest_local(path, data_dir, detector, family, manifest,
                         skip_empty=skip_empty,
                         frame_interval=frame_interval,
                         face_action=face_action)


def _ingest_local(
    path: Path,
    data_dir: Path,
    detector: Detector,
    family: str,
    manifest: list[ManifestEntry],
    *,
    skip_empty: bool = False,
    frame_interval: float = 2.0,
    face_action: str = "blur",
) -> int:
    if path.is_dir():
        print(f"Ingesting directory: {path}")
        return ingest_directory(path, data_dir, detector, family, path.name, manifest,
                                skip_empty=skip_empty)

    suffix = path.suffix.lower()
    if suffix == ".zip":
        print(f"Ingesting ZIP archive: {path.name}")
        return ingest_zip(path, data_dir, detector, family, manifest,
                          skip_empty=skip_empty)

    if suffix in {".tar", ".gz", ".bz2", ".xz", ".tgz"}:
        print(f"Ingesting TAR archive: {path.name}")
        return ingest_tar(path, data_dir, detector, family, manifest,
                          skip_empty=skip_empty)

    if is_video_file(path):
        print(f"Ingesting video: {path.name}")
        return ingest_video(
            path, data_dir, detector, family, manifest,
            skip_empty=skip_empty,
            frame_interval=frame_interval,
            face_action=face_action,
        )

    if is_image_file(path):
        print(f"Ingesting single image: {path.name}")
        result = process_single_image(
            path, data_dir, detector, family, path.name, manifest,
            skip_empty=skip_empty,
        )
        return 1 if result is not None else 0

    print(f"Skipping unsupported file: {path.name}")
    return 0
