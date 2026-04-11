"""Generate synthetic edge-case AprilTag images into a temp dir.

Writes a batch of scenario PNGs under a new /tmp/apriltag_synth_* dir and
prints copy-pasteable commands for running the normal ingest pipeline
against them and cleaning up afterwards. Does not touch ./data.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from apriltag_dataset import synthetic
from apriltag_dataset.synthetic import SCENARIOS


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="apriltag_synth_"))
    synth_dir = tmp / "synthetic"
    synth_dir.mkdir()

    for i, (name, fn) in enumerate(SCENARIOS):
        # Unique per-scenario seed drives a distinct low-frequency background
        # texture so dhash can't collapse near-uniform scenarios together.
        synthetic._SEED = i + 1
        out = synth_dir / f"synth_{i:02d}_{name}.png"
        fn(out)
        print(f"  wrote {out.name}")

    print(f"\nGenerated {len(SCENARIOS)} synthetic image(s) in {synth_dir}")
    print(f"Next: uv run python main.py ingest {synth_dir} --keep-empty")
    print(f"Then: rm -rf {tmp}")


if __name__ == "__main__":
    main()
