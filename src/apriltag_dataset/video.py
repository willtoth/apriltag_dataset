from __future__ import annotations

import re
import urllib.parse
from collections.abc import Generator
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}

_face_cascade: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_video_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""

    # YouTube
    if host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        return True
    if host == "youtu.be":
        return True

    # Google Drive file link
    if host in ("drive.google.com",) and "/file/d/" in parsed.path:
        return True

    return False


def is_playlist_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    qs = urllib.parse.parse_qs(parsed.query)

    if host in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        # /playlist?list=... is always a playlist
        if parsed.path == "/playlist" and "list" in qs:
            return True
        # /watch?v=...&list=... is a video within a playlist — treat as playlist
        if "list" in qs:
            return True

    return False


def iter_playlist_entries(url: str) -> Generator[tuple[int, int, str, str], None, None]:
    """Enumerate playlist videos without downloading.

    Yields (index, total, video_url, title) for each entry.
    """
    import yt_dlp

    print(f"Enumerating playlist: {url} ...")
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            print("  ERROR: Could not read playlist info")
            return

        entries = info.get("entries") or []
        entries = [e for e in entries if e is not None]
        total = len(entries)
        playlist_title = info.get("title", "playlist")
        print(f"  Playlist: {playlist_title} ({total} videos)")

        for i, entry in enumerate(entries):
            video_id = entry.get("id", entry.get("url", ""))
            title = entry.get("title", video_id)
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            yield i, total, video_url, title


def _extract_gdrive_file_id(url: str) -> str | None:
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def download_video(url: str, dest_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""

    # Google Drive
    if host == "drive.google.com":
        file_id = _extract_gdrive_file_id(url)
        if not file_id:
            raise ValueError(f"Could not extract file ID from Google Drive URL: {url}")
        import gdown

        dest = dest_dir / f"gdrive_{file_id}.mp4"
        print(f"Downloading from Google Drive (file ID: {file_id}) ...")
        gdown.download(id=file_id, output=str(dest), quiet=False)
        if not dest.exists():
            raise RuntimeError(f"gdown failed to download: {url}")
        return dest

    # YouTube and other yt-dlp supported sites
    import yt_dlp

    print(f"Downloading video: {url} ...")
    ydl_opts = {
        "format": (
            "bestvideo[height<=1080][vcodec~='^(avc|h264)']+bestaudio/"
            "bestvideo[height<=1080]+bestaudio/"
            "best[height<=1080]"
        ),
        "outtmpl": str(dest_dir / "%(title).60s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"yt-dlp returned no info for: {url}")
        filename = ydl.prepare_filename(info)
        # yt-dlp may have merged to mp4
        result_path = Path(filename)
        if not result_path.exists():
            result_path = result_path.with_suffix(".mp4")
        if not result_path.exists():
            # Search dest_dir for any video file
            for f in dest_dir.iterdir():
                if f.suffix.lower() in VIDEO_EXTENSIONS:
                    return f
            raise RuntimeError(f"Downloaded file not found for: {url}")
        return result_path


def detect_faces(grey_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        grey_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def blur_faces(
    grey_frame: np.ndarray, faces: list[tuple[int, int, int, int]]
) -> np.ndarray:
    result = grey_frame.copy()
    for x, y, w, h in faces:
        roi = result[y : y + h, x : x + w]
        # Use a large kernel for strong blur
        ksize = max(w, h) | 1  # ensure odd
        ksize = max(ksize, 51)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
        result[y : y + h, x : x + w] = blurred
    return result


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 2.0,
    face_action: str = "blur",
    dhash_threshold: int = 5,
) -> Generator[tuple[Path, str], None, None]:
    """Extract frames from a video, dedup sequentially, handle faces.

    Yields (frame_path, frame_label) tuples for frames that pass dedup and
    face filtering. The caller is responsible for deleting temp files.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path.name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"  ERROR: Cannot read FPS from video: {video_path.name}")
        cap.release()
        return

    frame_step = max(1, int(fps * interval_seconds))
    expected = total_frames // frame_step if frame_step > 0 else 0
    print(
        f"  Video: {video_path.name} ({total_frames} frames, {fps:.1f} fps, "
        f"extracting every {interval_seconds}s -> ~{expected} frames)"
    )

    prev_dhash: imagehash.ImageHash | None = None
    frame_idx = 0
    extracted = 0
    deduped = 0
    face_skipped = 0

    while True:
        target_frame = frame_idx * frame_step
        if target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            frame_idx += 1
            continue

        # Convert to greyscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        extracted += 1

        # Sequential dedup via dhash
        pil_img = Image.fromarray(grey)
        current_dhash = imagehash.dhash(pil_img)
        if prev_dhash is not None and (current_dhash - prev_dhash) <= dhash_threshold:
            frame_idx += 1
            deduped += 1
            continue
        prev_dhash = current_dhash

        # Face detection/handling
        if face_action != "none":
            faces = detect_faces(grey)
            if faces:
                if face_action == "skip":
                    frame_idx += 1
                    face_skipped += 1
                    continue
                elif face_action == "blur":
                    grey = blur_faces(grey, faces)

        # Save to temp file
        label = f"{video_path.stem}_frame_{target_frame:06d}"
        out_path = output_dir / f"{label}.png"
        cv2.imwrite(str(out_path), grey)

        frame_idx += 1
        yield out_path, label

    cap.release()

    kept = extracted - deduped - face_skipped
    parts = [f"Extracted {extracted} frames"]
    if deduped:
        parts.append(f"{deduped} deduped")
    if face_skipped:
        parts.append(f"{face_skipped} skipped (faces)")
    parts.append(f"{kept} passed to ingestion")
    print(f"  {', '.join(parts)}")
