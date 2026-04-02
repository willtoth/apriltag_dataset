# AprilTag Dataset

A ground-truth dataset of AprilTag detections. Images are ingested, preprocessed, and run through the upstream [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag) detector (via `pupil-apriltags`). Detection results are stored as JSON sidecars alongside the images.

Use this to validate new algorithm versions or train models against known-good outputs.

**Dataset on Hugging Face**: [wt200999/apriltag-dataset](https://huggingface.co/datasets/wt200999/apriltag-dataset)

## Quick start

```bash
# Load via Hugging Face datasets library
from datasets import load_dataset
ds = load_dataset("wt200999/apriltag-dataset")
```

Or clone and work locally:

```bash
git clone https://github.com/willtoth/apriltag-dataset
cd apriltag-dataset
uv sync
uv run python main.py download   # fetch images from Hugging Face
```

## Setup

Requires Python 3.13+.

```bash
uv sync
```

## Usage

All commands can be run via `uv run python main.py` or `uv run apriltag-dataset`.

### Ingest images

Add images from local files, directories, zip/tar archives, or URLs:

```bash
# Single image
uv run python main.py ingest photo.jpg

# From a URL
uv run python main.py ingest https://example.com/apriltag_photo.png

# ZIP or TAR archive
uv run python main.py ingest images.zip
uv run python main.py ingest images.tar.gz

# Directory of images
uv run python main.py ingest ./my_photos/

# Multiple sources at once
uv run python main.py ingest photo1.jpg photo2.png https://example.com/photo3.jpg
```

### Ingest videos

Extract frames from local video files, YouTube videos, Google Drive videos, or YouTube playlists. Frames go through the same preprocessing pipeline as images (grayscale, resize, dedup, detection). Faces are blurred by default for privacy.

```bash
# YouTube video
uv run python main.py ingest "https://www.youtube.com/watch?v=zwwqWDAw2JU"

# YouTube playlist (processes each video one at a time, deleting after each)
uv run python main.py ingest "https://www.youtube.com/playlist?list=PLPUJJPXRAlEQgL_Kp3YMQFmITfCzQrqe-"

# Google Drive video
uv run python main.py ingest "https://drive.google.com/file/d/1OFEerTz_puEJG8C3M6D15g2TPNTZb00F/view"

# Local video file
uv run python main.py ingest recording.mp4

# Custom frame interval (default: every 2 seconds)
uv run python main.py ingest video.mp4 --frame-interval 5

# Skip frames with faces instead of blurring them
uv run python main.py ingest video.mp4 --face-action skip

# Disable face detection entirely
uv run python main.py ingest video.mp4 --face-action none
```

### Ingest options

```
--family          Tag family to detect (default: tag36h11)
--nthreads        Detector threads (default: 4)
--quad-decimate   Quad decimation factor (default: 1.0, no decimation)
--frame-interval  Seconds between extracted video frames (default: 2.0)
--face-action     Face handling for video frames: blur (default), skip, or none
--data-dir        Data directory (default: ./data)
```

### Regenerate detections

Re-run detection on all existing images (e.g. after upgrading `pupil-apriltags` or changing the tag family):

```bash
uv run python main.py regenerate
uv run python main.py regenerate --family tagStandard41h12
```

### Rebuild manifest

Rebuild `manifest.json` from the existing images and detection files:

```bash
uv run python main.py manifest
```

### Dataset statistics

```bash
uv run python main.py stats
```

### Upload images to Hugging Face

After ingesting new images, push them to Hugging Face:

```bash
uv run python main.py upload
```

This regenerates `metadata.jsonl` and uploads all images.

### Download images from Hugging Face

```bash
uv run python main.py download
```

Downloads only images not already present locally.

### Repair detection metadata

One-time command to backfill `ingest_metadata` into detection sidecars and rebuild the manifest:

```bash
uv run python main.py repair
```

## Image preprocessing

During ingestion, every image is automatically:

1. Converted to **grayscale**
2. Resized to fit within **1920x1080** (preserving aspect ratio)
3. Saved as **lossless PNG**

## Deduplication

- **Exact duplicates** are detected by SHA-256 hash of the preprocessed image
- **Near-duplicates** are detected by perceptual hash (dhash, threshold: Hamming distance <= 5), unless the images differ by >25% in area (same scene at different scales is kept as useful test data)
- Images with **0 detections** can be re-ingested (e.g. with a different `--family`) to retry detection

## Dataset layout

```
data/
  images/           Preprocessed grayscale PNGs (content-hash prefixed)
  detections/       One JSON sidecar per image with detection results
  manifest.json     Dataset index
```

### Detection JSON format

Each sidecar in `data/detections/` contains:

```json
{
  "image_file": "a1b2c3d4e5f6_photo.png",
  "image_sha256": "a1b2c3d4e5f6...",
  "image_width": 1920,
  "image_height": 1080,
  "detector_config": {
    "library": "pupil-apriltags",
    "library_version": "1.0.4",
    "family": "tag36h11",
    "nthreads": 4,
    "quad_decimate": 1.0,
    "quad_sigma": 0.0,
    "refine_edges": true,
    "decode_sharpening": 0.25
  },
  "detected_at": "2026-03-24T12:00:00+00:00",
  "num_detections": 2,
  "detections": [
    {
      "tag_id": 42,
      "tag_family": "tag36h11",
      "hamming": 0,
      "decision_margin": 87.5,
      "center": [960.5, 540.2],
      "corners": [[920.1, 500.3], [1000.9, 500.1], [1001.2, 580.7], [919.8, 581.0]],
      "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
  ]
}
```

## Hosting

- **GitHub** has the code, detection JSONs, and manifest
- **Hugging Face** has the images and `metadata.jsonl` for the HF `datasets` library
- Images are gitignored locally; use `download`/`upload` to sync with HF

### Workflow

```
1. Ingest new images:      uv run python main.py ingest <source>
2. Commit to git:          git add -A && git commit -m "Add new detections"
3. Push detections:        git push
4. Upload images to HF:    uv run python main.py upload
```

## Supported tag families

`tag36h11` (default), `tagStandard41h12`, `tagStandard52h13`, `tagCircle21h7`, `tagCircle49h12`, `tagCustom48h12`, `tag25h9`, `tag16h5`
