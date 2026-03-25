# apriltag benchmark

Benchmarks the [apriltag](https://github.com/AprilRobotics/apriltag) C library against this dataset. Processes each image multiple times, measures wall-clock detection time, and verifies accuracy against ground truth.

## Prerequisites

- CMake 3.16+
- C compiler (GCC or Clang)
- Built apriltag library (source + `libapriltag.so`)
- Python 3 with `pandas` and `matplotlib` (for plotting only)

## Build

```bash
cd bench
mkdir build && cd build
cmake .. -DAPRILTAG_DIR=/path/to/apriltag
make
```

`APRILTAG_DIR` should point to the apriltag source root (containing `apriltag.h`). The built `libapriltag.so` is expected in `$APRILTAG_DIR/build/`.

## Run

```bash
# Full benchmark (all images, 10 iterations each)
./bench_apriltag -d ../../data

# Quick test with 50 images
./bench_apriltag -d ../../data --limit 50

# Custom configuration
./bench_apriltag -d ../../data -n 20 -w 2 -t 8 --decimate 2.0
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --data-dir` | `../data` | Dataset root directory |
| `-n, --iterations` | `10` | Total iterations per image |
| `-w, --warmup` | `1` | Warmup iterations (discarded from timing) |
| `-t, --threads` | `4` | Detector thread count |
| `-x, --decimate` | `1.0` | Quad decimation factor |
| `-b, --blur` | `0.0` | Gaussian blur sigma (negative sharpens) |
| `--refine-edges` | `1` | Refine tag edges (0 or 1) |
| `--sharpening` | `0.25` | Decode sharpening filter |
| `--hamming` | `2` | Max hamming bit errors to accept |
| `-o, --output` | `results.csv` | CSV output path |
| `-s, --summary` | `summary.json` | Summary JSON output path |
| `-q, --quiet` | off | Suppress per-image progress |
| `-l, --limit` | `0` | Only process first N images (0 = all) |
| `--no-verify` | off | Skip accuracy verification |

The default detector parameters match the ground truth configuration, so accuracy verification should report near-zero error. Changing detector parameters (e.g. `--decimate 2.0`) will likely cause accuracy to diverge from ground truth -- this is expected.

## Output

### CSV (`results.csv`)

One row per image with columns:

```
filename, width, height, resolution, gt_tags, det_tags,
tags_correct, tags_missed, tags_extra,
max_center_err_px, max_corner_err_px,
min_ms, max_ms, mean_ms, median_ms, stddev_ms
```

### Summary JSON (`summary.json`)

Aggregated results with the full detector config, per-resolution timing percentiles, and accuracy totals.

### Console

```
Apriltag Benchmark
  Images:     2762
  Iterations: 10 (1 warmup + 9 measured)
  Threads:    4
  Config:     decimate=1.0 sigma=0.0 refine=1 sharpening=0.25 hamming=1

[   1/2762] 0006f5b0a06c_PC00...7264f.png   640x360   tags= 2  median=4.96 ms
...

=== Summary by Resolution ===
Resolution      Count   Mean(ms) Median(ms)    P95(ms)    P99(ms)   Tags/img
640x360          2393      4.50      4.40       5.80       6.20       1.8
640x480            12      6.10      5.90       7.20       7.30       2.5
640x640           325      9.80      9.50      12.00      13.10       3.2
1080x1080          26     35.00     34.20      40.00      42.00       4.8
ALL              2762      6.20      4.60      12.00      35.00       2.1

=== Accuracy ===
Total GT tags:       4931
Total detected:      4932
Matched correctly:   4931 (100.00%)
Missed:                 0
Extra:                  1
Max center error:  0.0420 px
Max corner error:  0.0720 px
```

## Plots

Generate visualizations from the CSV:

```bash
pip install pandas matplotlib
python3 ../plot_results.py results.csv plots/
```

Produces:

| File | Description |
|------|-------------|
| `histograms_by_resolution.png` | Per-resolution histograms of detection time with median/P95 lines |
| `boxplot_by_resolution.png` | Side-by-side box plot comparing resolutions |
| `time_vs_tags.png` | Scatter plot of detection time vs tag count, colored by resolution |
| `accuracy_distribution.png` | Histograms of center and corner position error |
| `summary_table.png` | Summary statistics table as an image |

## How it works

1. Scans `data/images/` for PNG files, sorted alphabetically
2. For each image, loads the corresponding ground truth JSON from `data/detections/`
3. Loads the PNG into the apriltag `image_u8_t` format (grayscale, 96-byte stride alignment)
4. Runs warmup iterations (discarded), then measured iterations timed with `clock_gettime(CLOCK_MONOTONIC)`
5. Compares the last iteration's detections against ground truth by matching tag IDs and computing center/corner position error
6. Writes per-image results to CSV and prints a resolution-binned summary

Image loading (PNG decode) is **not** included in the timing -- only `apriltag_detector_detect()` is measured, which reflects the actual detection cost.
