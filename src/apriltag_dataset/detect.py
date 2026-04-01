from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pupil_apriltags import Detector

from .schema import DetectorConfig, ImageDetectionResult, TagDetection


def create_detector(
    family: str = "tag36h11",
    nthreads: int = 4,
    quad_decimate: float = 1.0,
    quad_sigma: float = 0.0,
    refine_edges: int = 1,
    decode_sharpening: float = 0.25,
) -> Detector:
    return Detector(
        families=family,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
    )


def get_detector_config(detector: Detector, family: str) -> DetectorConfig:
    import pupil_apriltags

    params = detector.params
    return DetectorConfig(
        library="pupil-apriltags",
        library_version=getattr(pupil_apriltags, "__version__", "unknown"),
        family=family,
        nthreads=params["nthreads"],
        quad_decimate=params["quad_decimate"],
        quad_sigma=params["quad_sigma"],
        refine_edges=bool(params["refine_edges"]),
        decode_sharpening=params["decode_sharpening"],
    )


def detect_image(
    detector: Detector,
    grey_image: Image.Image,
    image_filename: str,
    image_sha256: str,
    family: str,
) -> ImageDetectionResult:
    img_array = np.array(grey_image, dtype=np.uint8)
    raw_detections = detector.detect(img_array)

    detections = []
    for det in raw_detections:
        detections.append(
            TagDetection(
                tag_id=det.tag_id,
                tag_family=det.tag_family.decode()
                if isinstance(det.tag_family, bytes)
                else det.tag_family,
                hamming=det.hamming,
                decision_margin=float(det.decision_margin),
                center=det.center.tolist(),
                corners=det.corners.tolist(),
                homography=det.homography.tolist(),
            )
        )

    w, h = grey_image.size
    config = get_detector_config(detector, family)

    return ImageDetectionResult(
        image_file=image_filename,
        image_sha256=image_sha256,
        image_width=w,
        image_height=h,
        detector_config=config,
        detected_at=datetime.now(timezone.utc).isoformat(),
        num_detections=len(detections),
        detections=detections,
    )
