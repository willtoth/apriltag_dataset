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
class IngestMetadata:
    original_name: str
    source: str
    dhash: str
    ingested_at: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> IngestMetadata:
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
    ingest_metadata: IngestMetadata | None = None

    def to_dict(self) -> dict:
        d = {
            "image_file": self.image_file,
            "image_sha256": self.image_sha256,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detector_config": self.detector_config.to_dict(),
            "detected_at": self.detected_at,
            "num_detections": self.num_detections,
            "detections": [det.to_dict() for det in self.detections],
        }
        if self.ingest_metadata is not None:
            d["ingest_metadata"] = self.ingest_metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ImageDetectionResult:
        meta = None
        if "ingest_metadata" in d:
            meta = IngestMetadata.from_dict(d["ingest_metadata"])
        return cls(
            image_file=d["image_file"],
            image_sha256=d["image_sha256"],
            image_width=d["image_width"],
            image_height=d["image_height"],
            detector_config=DetectorConfig.from_dict(d["detector_config"]),
            detected_at=d["detected_at"],
            num_detections=d["num_detections"],
            detections=[TagDetection.from_dict(det) for det in d["detections"]],
            ingest_metadata=meta,
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
