"""Synthetic AprilTag scenario generator.

Writes extreme/edge-case PNGs designed to probe detector behavior. Pure
generator — does not touch the dataset or call the detector. Consumed by
``scripts/generate_synthetic.py``.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

_ASSETS_DIR = Path(__file__).parent / "synthetic_assets"
_BG = 200  # light-gray canvas fill

# Per-scenario seed; the generator script sets this before invoking each
# scenario so make_canvas can add a unique low-frequency texture to the
# background. Without this, every near-uniform scenario produces nearly
# identical dhashes and the ingest dedup layer collapses them together.
_SEED: int | None = None


def load_tag_bitmap(tag_id: int, family: str = "tag36h11") -> np.ndarray:
    path = _ASSETS_DIR / f"{family}_{tag_id:05d}.png"
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def render_tag(tag_id: int, size_px: int, family: str = "tag36h11") -> np.ndarray:
    bitmap = load_tag_bitmap(tag_id, family)
    return cv2.resize(bitmap, (size_px, size_px), interpolation=cv2.INTER_NEAREST)


def make_canvas(width: int, height: int, fill: int = _BG) -> np.ndarray:
    if _SEED is None:
        return np.full((height, width), fill, dtype=np.uint8)
    rng = np.random.default_rng(_SEED)
    coarse = rng.normal(0, 12, (16, 16)).astype(np.float32)
    texture = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(np.full((height, width), fill, dtype=np.float32) + texture,
                   0, 255).astype(np.uint8)


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    ch, cw = canvas.shape
    ph, pw = patch.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(cw, x + pw), min(ch, y + ph)
    if x0 >= x1 or y0 >= y1:
        return
    px0, py0 = x0 - x, y0 - y
    canvas[y0:y1, x0:x1] = patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]


def save_png(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img, mode="L").save(path, format="PNG")


def _centered(tag_size: int = 400, tag_id: int = 0, w: int = 1920, h: int = 1080) -> np.ndarray:
    canvas = make_canvas(w, h)
    tag = render_tag(tag_id, tag_size)
    paste(canvas, tag, (w - tag_size) // 2, (h - tag_size) // 2)
    return canvas


def warp_yaw(tag: np.ndarray, yaw_deg: float) -> np.ndarray:
    h, w = tag.shape
    far = h * math.cos(math.radians(yaw_deg))
    y_off = (h - far) / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, y_off], [w, y_off + far], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(tag, M, (w, h), borderValue=_BG)


def rotate_padded(tag: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = tag.shape
    diag = int(math.ceil(math.sqrt(h * h + w * w)))
    padded = np.full((diag, diag), _BG, dtype=np.uint8)
    off = (diag - w) // 2
    padded[off:off + h, off:off + w] = tag
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle_deg, 1.0)
    return cv2.warpAffine(padded, M, (diag, diag), borderValue=_BG)


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# A. Size extremes
# ---------------------------------------------------------------------------

def _scn_tiny_tag_24px(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    paste(canvas, render_tag(0, 24), (1920 - 24) // 2, (1080 - 24) // 2)
    save_png(out, canvas)


def _scn_ultra_tiny_tag_12px(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    paste(canvas, render_tag(0, 12), (1920 - 12) // 2, (1080 - 12) // 2)
    save_png(out, canvas)


def _scn_huge_tag_90pct(out: Path) -> None:
    size = 972
    canvas = make_canvas(1920, 1080)
    paste(canvas, render_tag(0, size), (1920 - size) // 2, (1080 - size) // 2)
    save_png(out, canvas)


def _scn_massive_tag_full_frame(out: Path) -> None:
    # 10x10 bitmap → crop to inner 8x8 (strip quiet zone) → upscale to canvas
    inner = load_tag_bitmap(0)[1:9, 1:9]
    canvas = cv2.resize(inner, (1080, 1080), interpolation=cv2.INTER_NEAREST)
    save_png(out, canvas)


def _scn_oversized_input_4k(out: Path) -> None:
    canvas = make_canvas(4096, 3000)
    paste(canvas, render_tag(0, 200), (4096 - 200) // 2, (3000 - 200) // 2)
    save_png(out, canvas)


def _scn_huge_canvas_8k(out: Path) -> None:
    canvas = make_canvas(7680, 4320)
    paste(canvas, render_tag(0, 400), (7680 - 400) // 2, (4320 - 400) // 2)
    save_png(out, canvas)


# ---------------------------------------------------------------------------
# B. Canvas-shape extremes
# ---------------------------------------------------------------------------

def _scn_nano_canvas_64x64(out: Path) -> None:
    canvas = make_canvas(64, 64)
    paste(canvas, render_tag(0, 24), 20, 20)
    save_png(out, canvas)


def _scn_undersized_input_320(out: Path) -> None:
    canvas = make_canvas(320, 240)
    paste(canvas, render_tag(0, 60), (320 - 60) // 2, (240 - 60) // 2)
    save_png(out, canvas)


def _scn_cinema_ultrawide_8000x640(out: Path) -> None:
    canvas = make_canvas(8000, 640)
    paste(canvas, render_tag(0, 80), 8000 - 200, (640 - 80) // 2)
    save_png(out, canvas)


def _scn_nonstandard_dims_1333x777(out: Path) -> None:
    canvas = make_canvas(1333, 777)
    paste(canvas, render_tag(0, 150), (1333 - 150) // 2, (777 - 150) // 2)
    save_png(out, canvas)


# ---------------------------------------------------------------------------
# C. Count extremes
# ---------------------------------------------------------------------------

def _scn_many_tags_grid_20(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    tag_size = 150
    rows, cols = 4, 5
    gap_x = (1920 - cols * tag_size) // (cols + 1)
    gap_y = (1080 - rows * tag_size) // (rows + 1)
    ids = [0, 1, 5, 10, 42]
    i = 0
    for r in range(rows):
        for c in range(cols):
            x = gap_x + c * (tag_size + gap_x)
            y = gap_y + r * (tag_size + gap_y)
            paste(canvas, render_tag(ids[i % len(ids)], tag_size), x, y)
            i += 1
    save_png(out, canvas)


def _scn_hundred_tags_chaos(out: Path) -> None:
    rng = np.random.default_rng(42)
    canvas = make_canvas(1920, 1080)
    ids = [0, 1, 5, 10, 42]
    for _ in range(100):
        size = int(rng.integers(30, 80))
        tag = render_tag(int(rng.choice(ids)), size)
        rotated = rotate_padded(tag, float(rng.uniform(0, 360)))
        rh, rw = rotated.shape
        x = int(rng.integers(0, max(1, 1920 - rw)))
        y = int(rng.integers(0, max(1, 1080 - rh)))
        paste(canvas, rotated, x, y)
    save_png(out, canvas)


def _scn_single_tag_near_border(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    paste(canvas, render_tag(0, 300), -90, -90)
    save_png(out, canvas)


# ---------------------------------------------------------------------------
# D. Geometric difficulty
# ---------------------------------------------------------------------------

def _scn_perspective_steep_75(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    warped = warp_yaw(render_tag(0, 400), 75.0)
    paste(canvas, warped, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_perspective_extreme_85(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    warped = warp_yaw(render_tag(0, 400), 85.0)
    paste(canvas, warped, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_in_plane_rotation_23(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    r = rotate_padded(render_tag(0, 400), 23.0)
    paste(canvas, r, (1920 - r.shape[1]) // 2, (1080 - r.shape[0]) // 2)
    save_png(out, canvas)


def _scn_in_plane_rotation_45(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    r = rotate_padded(render_tag(0, 400), 45.0)
    paste(canvas, r, (1920 - r.shape[1]) // 2, (1080 - r.shape[0]) // 2)
    save_png(out, canvas)


def _scn_in_plane_rotation_180(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    r = rotate_padded(render_tag(0, 400), 180.0)
    paste(canvas, r, (1920 - r.shape[1]) // 2, (1080 - r.shape[0]) // 2)
    save_png(out, canvas)


def _scn_shear_heavy(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    tag = render_tag(0, 400)
    h, w = tag.shape
    shear = 0.6
    new_w = int(w + shear * h)
    M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(tag, M, (new_w, h), borderValue=_BG)
    paste(canvas, sheared, (1920 - new_w) // 2, (1080 - h) // 2)
    save_png(out, canvas)


def _scn_nonuniform_scale_3x1(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    bitmap = load_tag_bitmap(0)
    stretched = cv2.resize(bitmap, (900, 300), interpolation=cv2.INTER_NEAREST)
    paste(canvas, stretched, (1920 - 900) // 2, (1080 - 300) // 2)
    save_png(out, canvas)


def _scn_half_off_frame(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    paste(canvas, render_tag(0, 400), 1920 - 200, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_barrel_distortion(out: Path) -> None:
    canvas = _centered(500)
    h, w = canvas.shape
    cy, cx = h / 2, w / 2
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    factor = 1 + 6e-7 * r2
    src_x = (cx + dx * factor).astype(np.float32)
    src_y = (cy + dy * factor).astype(np.float32)
    distorted = cv2.remap(canvas, src_x, src_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=_BG)
    save_png(out, distorted)


def _scn_mirror_flipped_horizontal(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    mirrored = np.fliplr(render_tag(0, 400))
    paste(canvas, mirrored, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


# ---------------------------------------------------------------------------
# E. Photometric difficulty
# ---------------------------------------------------------------------------

def _motion_blur(img: np.ndarray, length: int) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(img, -1, kernel)


def _scn_motion_blur_15px(out: Path) -> None:
    save_png(out, _motion_blur(_centered(400), 15))


def _scn_motion_blur_extreme_40px(out: Path) -> None:
    save_png(out, _motion_blur(_centered(400), 40))


def _scn_defocus_blur_sigma3(out: Path) -> None:
    save_png(out, cv2.GaussianBlur(_centered(400), (0, 0), sigmaX=3.0))


def _scn_defocus_blur_sigma10(out: Path) -> None:
    save_png(out, cv2.GaussianBlur(_centered(400), (0, 0), sigmaX=10.0))


def _scn_low_contrast_40_200(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    tag = render_tag(0, 400)
    tag_lc = (tag.astype(np.float32) * (200 - 40) / 255 + 40).astype(np.uint8)
    paste(canvas, tag_lc, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_jpeg_quality_15(out: Path) -> None:
    save_png(out, jpeg_roundtrip(_centered(400), 15))


def _scn_jpeg_quality_3(out: Path) -> None:
    save_png(out, jpeg_roundtrip(_centered(400), 3))


def _scn_gradient_lighting(out: Path) -> None:
    canvas = _centered(400)
    grad = np.linspace(0.4, 1.4, 1920, dtype=np.float32)[None, :]
    out_img = np.clip(canvas.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    save_png(out, out_img)


def _scn_specular_highlight(out: Path) -> None:
    canvas = _centered(400).astype(np.float32)
    h, w = canvas.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, sigma = 1060.0, 640.0, 80.0
    spot = 200.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
    save_png(out, np.clip(canvas + spot, 0, 255).astype(np.uint8))


def _scn_gaussian_noise_sigma25(out: Path) -> None:
    rng = np.random.default_rng(0)
    canvas = _centered(400).astype(np.float32)
    canvas += rng.normal(0, 25, canvas.shape).astype(np.float32)
    save_png(out, np.clip(canvas, 0, 255).astype(np.uint8))


def _scn_salt_pepper_10pct(out: Path) -> None:
    rng = np.random.default_rng(1)
    canvas = _centered(400)
    mask = rng.random(canvas.shape)
    canvas[mask < 0.05] = 0
    canvas[mask >= 0.95] = 255
    save_png(out, canvas)


def _scn_overexposed_clipped(out: Path) -> None:
    canvas = _centered(400).astype(np.float32)
    canvas = canvas / 255.0 * 25.0 + 230.0
    save_png(out, canvas.astype(np.uint8))


def _scn_underexposed_crushed(out: Path) -> None:
    canvas = _centered(400).astype(np.float32)
    canvas = canvas / 255.0 * 25.0
    save_png(out, canvas.astype(np.uint8))


def _scn_negative_inverted(out: Path) -> None:
    save_png(out, 255 - _centered(400))


def _scn_bimodal_lighting(out: Path) -> None:
    canvas = _centered(400).astype(np.float32)
    canvas[:, :960] *= 0.3
    canvas[:, 960:] = np.clip(canvas[:, 960:] * 1.3, 0, 255)
    save_png(out, canvas.astype(np.uint8))


# ---------------------------------------------------------------------------
# F. Occlusion / overlay
# ---------------------------------------------------------------------------

def _scn_center_occluded_square(out: Path) -> None:
    canvas = _centered(400)
    canvas[480:600, 900:1020] = 0
    save_png(out, canvas)


def _scn_diagonal_line_scratch(out: Path) -> None:
    canvas = _centered(400)
    cv2.line(canvas, (700, 300), (1200, 800), color=0, thickness=15)
    save_png(out, canvas)


def _scn_text_overlay(out: Path) -> None:
    canvas = _centered(400)
    cv2.putText(canvas, "FRC2026", (720, 580), cv2.FONT_HERSHEY_SIMPLEX,
                4.0, 0, 12, cv2.LINE_AA)
    save_png(out, canvas)


def _scn_reflection_ghost(out: Path) -> None:
    base = _centered(400)
    ghost = make_canvas(1920, 1080)
    paste(ghost, render_tag(0, 400), (1920 - 400) // 2 + 60, (1080 - 400) // 2 + 60)
    mixed = base.astype(np.float32) * 0.6 + ghost.astype(np.float32) * 0.4
    save_png(out, mixed.astype(np.uint8))


# ---------------------------------------------------------------------------
# G. Negative / decoy cases
# ---------------------------------------------------------------------------

def _scn_pure_gray_118(out: Path) -> None:
    save_png(out, make_canvas(1920, 1080, fill=118))


def _scn_all_black(out: Path) -> None:
    save_png(out, make_canvas(1920, 1080, fill=0))


def _scn_all_white(out: Path) -> None:
    save_png(out, make_canvas(1920, 1080, fill=255))


def _scn_random_uniform_noise(out: Path) -> None:
    rng = np.random.default_rng(7)
    save_png(out, rng.integers(0, 256, (1080, 1920), dtype=np.uint8))


def _scn_checkerboard_decoy(out: Path) -> None:
    h, w = 1080, 1920
    cell = 60
    y, x = np.mgrid[0:h, 0:w]
    checker = (((x // cell) + (y // cell)) % 2 == 0).astype(np.float32) * 255.0
    checker += np.sin(x.astype(np.float32) / 50.0) * 10.0
    save_png(out, np.clip(checker, 0, 255).astype(np.uint8))


def _scn_qr_code_decoy(out: Path) -> None:
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 2, (21, 21), dtype=np.uint8) * 255
    tag = cv2.resize(blocks, (400, 400), interpolation=cv2.INTER_NEAREST)
    canvas = make_canvas(1920, 1080)
    paste(canvas, tag, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_wrong_family_tag25h9(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    tag = render_tag(0, 400, family="tag25h9")
    paste(canvas, tag, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


def _scn_wrong_family_standard41h12(out: Path) -> None:
    canvas = make_canvas(1920, 1080)
    tag = render_tag(0, 400, family="tagStandard41h12")
    paste(canvas, tag, (1920 - 400) // 2, (1080 - 400) // 2)
    save_png(out, canvas)


# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, Callable[[Path], None]]] = [
    # A. Size extremes
    ("tiny_tag_24px", _scn_tiny_tag_24px),
    ("ultra_tiny_tag_12px", _scn_ultra_tiny_tag_12px),
    ("huge_tag_90pct", _scn_huge_tag_90pct),
    ("massive_tag_full_frame", _scn_massive_tag_full_frame),
    ("oversized_input_4k", _scn_oversized_input_4k),
    ("huge_canvas_8k", _scn_huge_canvas_8k),
    # B. Canvas-shape extremes
    ("nano_canvas_64x64", _scn_nano_canvas_64x64),
    ("undersized_input_320", _scn_undersized_input_320),
    ("cinema_ultrawide_8000x640", _scn_cinema_ultrawide_8000x640),
    ("nonstandard_dims_1333x777", _scn_nonstandard_dims_1333x777),
    # C. Count extremes
    ("many_tags_grid_20", _scn_many_tags_grid_20),
    ("hundred_tags_chaos", _scn_hundred_tags_chaos),
    ("single_tag_near_border", _scn_single_tag_near_border),
    # D. Geometric difficulty
    ("perspective_steep_75", _scn_perspective_steep_75),
    ("perspective_extreme_85", _scn_perspective_extreme_85),
    ("in_plane_rotation_23", _scn_in_plane_rotation_23),
    ("in_plane_rotation_45", _scn_in_plane_rotation_45),
    ("in_plane_rotation_180", _scn_in_plane_rotation_180),
    ("shear_heavy", _scn_shear_heavy),
    ("nonuniform_scale_3x1", _scn_nonuniform_scale_3x1),
    ("half_off_frame", _scn_half_off_frame),
    ("barrel_distortion", _scn_barrel_distortion),
    ("mirror_flipped_horizontal", _scn_mirror_flipped_horizontal),
    # E. Photometric difficulty
    ("motion_blur_15px", _scn_motion_blur_15px),
    ("motion_blur_extreme_40px", _scn_motion_blur_extreme_40px),
    ("defocus_blur_sigma3", _scn_defocus_blur_sigma3),
    ("defocus_blur_sigma10", _scn_defocus_blur_sigma10),
    ("low_contrast_40_200", _scn_low_contrast_40_200),
    ("jpeg_quality_15", _scn_jpeg_quality_15),
    ("jpeg_quality_3", _scn_jpeg_quality_3),
    ("gradient_lighting", _scn_gradient_lighting),
    ("specular_highlight", _scn_specular_highlight),
    ("gaussian_noise_sigma25", _scn_gaussian_noise_sigma25),
    ("salt_pepper_10pct", _scn_salt_pepper_10pct),
    ("overexposed_clipped", _scn_overexposed_clipped),
    ("underexposed_crushed", _scn_underexposed_crushed),
    ("negative_inverted", _scn_negative_inverted),
    ("bimodal_lighting", _scn_bimodal_lighting),
    # F. Occlusion / overlay
    ("center_occluded_square", _scn_center_occluded_square),
    ("diagonal_line_scratch", _scn_diagonal_line_scratch),
    ("text_overlay", _scn_text_overlay),
    ("reflection_ghost", _scn_reflection_ghost),
    # G. Negative / decoy
    ("pure_gray_118", _scn_pure_gray_118),
    ("all_black", _scn_all_black),
    ("all_white", _scn_all_white),
    ("random_uniform_noise", _scn_random_uniform_noise),
    ("checkerboard_decoy", _scn_checkerboard_decoy),
    ("qr_code_decoy", _scn_qr_code_decoy),
    ("wrong_family_tag25h9", _scn_wrong_family_tag25h9),
    ("wrong_family_standard41h12", _scn_wrong_family_standard41h12),
]
