"""Shared, dependency-free data models for football vision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned box in image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding-box maxima must not precede minima")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    def iou(self, other: BoundingBox) -> float:
        left, top = max(self.x1, other.x1), max(self.y1, other.y1)
        right, bottom = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_name: str
    appearance: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")
        if not self.class_name:
            raise ValueError("Detection class_name must not be empty")
        if self.appearance is not None and not all(
            isfinite(value) for value in self.appearance
        ):
            raise ValueError("Appearance features must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "class_name": self.class_name,
            "appearance": list(self.appearance) if self.appearance is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class Track:
    track_id: int
    bbox: BoundingBox
    class_name: str
    confidence: float
    age: int = 1
    hits: int = 1
    missed: int = 0
    last_frame: int = 0
    appearance: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_list(),
            "class_name": self.class_name,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
            "missed": self.missed,
            "last_frame": self.last_frame,
            "appearance": list(self.appearance) if self.appearance is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection]
    tracks: list[Track]
    ball_assignment: dict[str, Any] | None = None
    camera_motion: dict[str, Any] | None = None
    violations: list[str] = field(default_factory=list)
    homography: dict[str, Any] | None = None
    ball_trajectory: dict[str, Any] | None = None
    possession: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "frame_index": self.frame_index,
            "detections": [detection.to_dict() for detection in self.detections],
            "tracks": [track.to_dict() for track in self.tracks],
            "ball_assignment": self.ball_assignment,
            "camera_motion": self.camera_motion,
            "violations": list(self.violations),
            "events": list(self.events),
        }
        if self.possession is not None:
            result["possession"] = self.possession
        if self.homography is not None:
            result["homography"] = self.homography
        if self.ball_trajectory is not None:
            result["ball_trajectory"] = self.ball_trajectory
        return result
