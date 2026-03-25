#!/usr/bin/env python3
"""
Generate benchmark visualization plots from bench_apriltag CSV output.

Usage: python3 plot_results.py results.csv [output_dir]
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_histograms_by_resolution(df, output_dir):
    """Per-resolution histograms of detection time."""
    resolutions = sorted(
        df["resolution"].unique(),
        key=lambda r: df[df.resolution == r]["median_ms"].median(),
    )

    n = len(resolutions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), squeeze=False)
    axes = axes.flatten()

    for ax, res in zip(axes, resolutions):
        subset = df[df.resolution == res]["median_ms"]
        bins = min(50, max(10, len(subset) // 5))
        ax.hist(subset, bins=bins, alpha=0.7, edgecolor="black", color="#4C72B0")
        med = subset.median()
        p95 = subset.quantile(0.95)
        ax.axvline(med, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median: {med:.2f} ms")
        ax.axvline(p95, color="orange", linestyle="--", linewidth=1.5,
                   label=f"P95: {p95:.2f} ms")
        ax.set_title(f"{res}  (n={len(subset)})", fontsize=12)
        ax.set_xlabel("Detection Time (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "histograms_by_resolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_boxplot_by_resolution(df, output_dir):
    """Box plot comparing detection time across resolutions."""
    resolutions = sorted(
        df["resolution"].unique(),
        key=lambda r: df[df.resolution == r]["median_ms"].median(),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[df.resolution == r]["median_ms"].values for r in resolutions]
    labels = [f"{r}\n(n={len(df[df.resolution==r])})" for r in resolutions]

    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)

    ax.set_ylabel("Detection Time (ms)")
    ax.set_title("Detection Time by Resolution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_by_resolution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_time_vs_tags(df, output_dir):
    """Scatter plot of detection time vs number of tags detected."""
    fig, ax = plt.subplots(figsize=(10, 6))

    resolutions = sorted(
        df["resolution"].unique(),
        key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1]),
    )
    colors = plt.cm.tab10(np.linspace(0, 1, len(resolutions)))

    for res, color in zip(resolutions, colors):
        subset = df[df.resolution == res]
        ax.scatter(
            subset["det_tags"], subset["median_ms"],
            alpha=0.4, label=res, s=15, color=color,
        )

    ax.set_xlabel("Number of Tags Detected")
    ax.set_ylabel("Detection Time (ms)")
    ax.set_title("Detection Time vs Number of Tags")
    ax.legend(title="Resolution")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "time_vs_tags.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_accuracy(df, output_dir):
    """Accuracy error distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    center_err = df["max_center_err_px"]
    corner_err = df["max_corner_err_px"]

    ax1.hist(center_err, bins=50, alpha=0.7, edgecolor="black", color="#55A868")
    ax1.set_xlabel("Max Center Error (px)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Center Position Error Distribution\n"
                  f"max={center_err.max():.4f} px, mean={center_err.mean():.4f} px")

    ax2.hist(corner_err, bins=50, alpha=0.7, edgecolor="black", color="#C44E52")
    ax2.set_xlabel("Max Corner Error (px)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Corner Position Error Distribution\n"
                  f"max={corner_err.max():.4f} px, mean={corner_err.mean():.4f} px")

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {path}")


def plot_summary_table(df, output_dir):
    """Summary statistics table rendered as a figure."""
    resolutions = sorted(
        df["resolution"].unique(),
        key=lambda r: int(r.split("x")[0]) * int(r.split("x")[1]),
    )

    headers = ["Resolution", "Count", "Mean (ms)", "Median (ms)",
               "P95 (ms)", "P99 (ms)", "Avg Tags"]
    rows = []

    for res in resolutions:
        s = df[df.resolution == res]["median_ms"]
        tags = df[df.resolution == res]["det_tags"]
        rows.append([
            res, str(len(s)),
            f"{s.mean():.2f}", f"{s.median():.2f}",
            f"{s.quantile(0.95):.2f}", f"{s.quantile(0.99):.2f}",
            f"{tags.mean():.1f}",
        ])

    # ALL row
    s = df["median_ms"]
    tags = df["det_tags"]
    rows.append([
        "ALL", str(len(s)),
        f"{s.mean():.2f}", f"{s.median():.2f}",
        f"{s.quantile(0.95):.2f}", f"{s.quantile(0.99):.2f}",
        f"{tags.mean():.1f}",
    ])

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style ALL row
    last_row = len(rows)
    for j in range(len(headers)):
        table[last_row, j].set_facecolor("#E8E8E8")
        table[last_row, j].set_text_props(fontweight="bold")

    plt.title("Benchmark Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    # Accuracy summary
    total_gt = df["gt_tags"].sum()
    total_correct = df["tags_correct"].sum()
    total_missed = df["tags_missed"].sum()
    total_extra = df["tags_extra"].sum()
    print(f"\nAccuracy: {total_correct}/{total_gt} matched "
          f"({100*total_correct/total_gt:.2f}%), "
          f"{total_missed} missed, {total_extra} extra")
    print(f"Max center error: {df['max_center_err_px'].max():.4f} px")
    print(f"Max corner error: {df['max_corner_err_px'].max():.4f} px")
    print()

    print("Generating plots:")
    plot_histograms_by_resolution(df, output_dir)
    plot_boxplot_by_resolution(df, output_dir)
    plot_time_vs_tags(df, output_dir)
    plot_accuracy(df, output_dir)
    plot_summary_table(df, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
