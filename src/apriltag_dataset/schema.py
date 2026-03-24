from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class DetectorConfig:
    library: str
    library_version: str
    family: str
    nthreads: int
    quad_decimate: float
    quad_sigma: float
    refine_edges: bool
    decode_sharpening: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DetectorConfig:
        return cls(**d)


@dataclass
class TagDetection:
    tag_id: int
    tag_family: str
    hamming: int
    decision_margin: float
    center: list[float]
    corners: list[list[float]]
    homography: list[list[float]]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TagDetection:
        return cls(**d)


@dataclass
class ImageDetectionResult:
    image_file: str
    image_sha256: str
    image_width: int
    image_height: int
    detector_config: DetectorConfig
    detected_at: str
    num_detections: int
    detections: list[TagDetection]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ImageDetectionResult:
        return cls(
            image_file=d["image_file"],
            image_sha256=d["image_sha256"],
            image_width=d["image_width"],
            image_height=d["image_height"],
            detector_config=DetectorConfig.from_dict(d["detector_config"]),
            detected_at=d["detected_at"],
            num_detections=d["num_detections"],
            detections=[TagDetection.from_dict(det) for det in d["detections"]],
        )


@dataclass
class ManifestEntry:
    filename: str
    original_name: str
    source: str
    sha256: str
    dhash: str
    width: int
    height: int
    ingested_at: str
    num_detections: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ManifestEntry:
        return cls(**d)
