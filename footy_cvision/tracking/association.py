"""Deterministic greedy association combining geometry and appearance."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from footy_cvision.core import BoundingBox, Detection, Track


@dataclass(frozen=True)
class AssociationConfig:
    iou_weight: float = 0.45
    center_weight: float = 0.35
    appearance_weight: float = 0.20
    max_center_distance: float = 100.0
    min_iou: float = 0.0
    max_cost: float = 0.95

    def __post_init__(self) -> None:
        if min(self.iou_weight, self.center_weight, self.appearance_weight) < 0:
            raise ValueError("Association weights must be non-negative")
        if self.iou_weight + self.center_weight + self.appearance_weight <= 0:
            raise ValueError("At least one association weight must be positive")
        if self.max_center_distance <= 0 or not 0 <= self.min_iou <= 1:
            raise ValueError("Invalid association gate")
        if self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")


def _appearance_distance(
    left: tuple[float, ...] | None, right: tuple[float, ...] | None
) -> float | None:
    if left is None or right is None or len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    cosine_similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return (1.0 - cosine_similarity) / 2.0


def _cost(
    bbox: BoundingBox,
    appearance: tuple[float, ...] | None,
    detection: Detection,
    config: AssociationConfig,
    max_distance: float,
) -> float | None:
    distance = sqrt(
        (bbox.center[0] - detection.bbox.center[0]) ** 2
        + (bbox.center[1] - detection.bbox.center[1]) ** 2
    )
    iou = bbox.iou(detection.bbox)
    if distance > max_distance or iou < config.min_iou:
        return None
    center_cost = min(1.0, distance / max_distance)
    app_cost = _appearance_distance(appearance, detection.appearance)
    weighted = [
        (1.0 - iou, config.iou_weight),
        (center_cost, config.center_weight),
    ]
    if app_cost is not None:
        weighted.append((app_cost, config.appearance_weight))
    weight_sum = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / weight_sum


def transform_bbox(
    bbox: BoundingBox, transform: Sequence[Sequence[float]]
) -> BoundingBox:
    if len(transform) != 2 or any(len(row) != 3 for row in transform):
        raise ValueError("Camera transform must be a 2x3 affine matrix")
    corners = (
        (bbox.x1, bbox.y1),
        (bbox.x1, bbox.y2),
        (bbox.x2, bbox.y1),
        (bbox.x2, bbox.y2),
    )
    points = [
        (
            float(transform[0][0]) * x
            + float(transform[0][1]) * y
            + float(transform[0][2]),
            float(transform[1][0]) * x
            + float(transform[1][1]) * y
            + float(transform[1][2]),
        )
        for x, y in corners
    ]
    return BoundingBox(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def associate(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    config: AssociationConfig | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match lowest-cost (track index, detection index) pairs."""
    config = config or AssociationConfig()
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            if track.class_name != detection.class_name:
                continue
            # Lost tracks receive a wider geometric gate for short occlusions.
            gate = config.max_center_distance * (1.0 + min(track.missed, 5) * 0.25)
            cost = _cost(
                track.bbox,
                track.appearance,
                detection,
                config,
                gate,
            )
            if cost is not None and cost <= config.max_cost:
                candidates.append((cost, track.track_id, detection_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    pairs: list[tuple[int, int]] = []
    track_index_by_id = {track.track_id: index for index, track in enumerate(tracks)}
    for _, track_id, detection_index in candidates:
        track_index = track_index_by_id[track_id]
        if track_index in matched_tracks or detection_index in matched_detections:
            continue
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)
        pairs.append((track_index, detection_index))
    pairs.sort()
    unmatched_tracks = [
        index for index in range(len(tracks)) if index not in matched_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections)) if index not in matched_detections
    ]
    return pairs, unmatched_tracks, unmatched_detections
