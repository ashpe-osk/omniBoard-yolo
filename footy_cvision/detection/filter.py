"""Confidence, class, and optional pitch-boundary filtering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import hypot, sqrt

from footy_cvision.core import BoundingBox, Detection

Point = tuple[float, float]
PERSON_CLASSES = {"player", "goalkeeper", "referee"}


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(
            y1, y2
        ) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def intersection_over_smaller_box(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return intersection area divided by the area of the smaller box."""
    left = max(box_a.x1, box_b.x1)
    top = max(box_a.y1, box_b.y1)
    right = min(box_a.x2, box_b.x2)
    bottom = min(box_a.y2, box_b.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller_area = min(box_a.area, box_b.area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


class DetectionFilter:
    def __init__(
        self,
        min_confidence: float = 0.0,
        allowed_classes: set[str] | None = None,
        excluded_classes: set[str] | None = None,
        pitch_polygon: Sequence[Point] | None = None,
        filter_out_of_bounds: bool = False,
        nms_iou_threshold: float | None = 0.5,
        containment_threshold: float | None = 0.90,
        person_duplicate_containment_threshold: float | None = 0.75,
        person_duplicate_footpoint_distance_ratio: float | None = 0.20,
        person_duplicate_center_distance_ratio: float | None = 0.50,
        person_duplicate_min_height_ratio: float | None = 0.55,
        person_duplicate_max_appearance_distance: float | None = 0.15,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if nms_iou_threshold is not None and not 0.0 < nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in (0, 1] or None")
        if containment_threshold is not None and not (
            0.0 < containment_threshold <= 1.0
        ):
            raise ValueError("containment_threshold must be in (0, 1] or None")
        if person_duplicate_containment_threshold is not None and not (
            0.0 < person_duplicate_containment_threshold <= 1.0
        ):
            raise ValueError(
                "person_duplicate_containment_threshold must be in (0, 1] or None"
            )
        if person_duplicate_footpoint_distance_ratio is not None and not (
            0.0 < person_duplicate_footpoint_distance_ratio <= 1.0
        ):
            raise ValueError(
                "person_duplicate_footpoint_distance_ratio must be in (0, 1] or None"
            )
        if person_duplicate_center_distance_ratio is not None and not (
            0.0 < person_duplicate_center_distance_ratio <= 1.0
        ):
            raise ValueError(
                "person_duplicate_center_distance_ratio must be in (0, 1]"
            )
        if person_duplicate_min_height_ratio is not None and not (
            0.0 < person_duplicate_min_height_ratio <= 1.0
        ):
            raise ValueError(
                "person_duplicate_min_height_ratio must be in (0, 1]"
            )
        if person_duplicate_max_appearance_distance is not None and not (
            0.0 <= person_duplicate_max_appearance_distance <= 1.0
        ):
            raise ValueError(
                "person_duplicate_max_appearance_distance must be in [0, 1]"
            )
        if filter_out_of_bounds and (
            pitch_polygon is None or len(pitch_polygon) < 3
        ):
            raise ValueError("A pitch polygon is required for bounds filtering")
        self.min_confidence = min_confidence
        self.allowed_classes = allowed_classes
        self.excluded_classes = excluded_classes or set()
        self.pitch_polygon = pitch_polygon
        self.filter_out_of_bounds = filter_out_of_bounds
        self.nms_iou_threshold = nms_iou_threshold
        self.containment_threshold = containment_threshold
        self.person_duplicate_containment_threshold = (
            person_duplicate_containment_threshold
        )
        self.person_duplicate_footpoint_distance_ratio = (
            person_duplicate_footpoint_distance_ratio
        )
        self.person_duplicate_center_distance_ratio = (
            person_duplicate_center_distance_ratio
        )
        self.person_duplicate_min_height_ratio = (
            person_duplicate_min_height_ratio
        )
        self.person_duplicate_max_appearance_distance = (
            person_duplicate_max_appearance_distance
        )

    def __call__(self, detections: Iterable[Detection]) -> list[Detection]:
        return self.filter(detections)

    def filter(self, detections: Iterable[Detection]) -> list[Detection]:
        candidates: list[Detection] = []
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            if self.allowed_classes is not None and (
                detection.class_name not in self.allowed_classes
            ):
                continue
            if detection.class_name in self.excluded_classes:
                continue
            if self.filter_out_of_bounds:
                assert self.pitch_polygon is not None
                if not point_in_polygon(
                    detection.bbox.footpoint, self.pitch_polygon
                ):
                    continue
            candidates.append(detection)
        if (
            self.nms_iou_threshold is None
            and self.containment_threshold is None
            and self.person_duplicate_containment_threshold is None
        ):
            return candidates
        result: list[Detection] = []
        for detection in sorted(
            candidates, key=lambda item: item.confidence, reverse=True
        ):
            if any(
                detection.class_name == kept.class_name
                and (
                    (
                        self.nms_iou_threshold is not None
                        and detection.bbox.iou(kept.bbox)
                        >= self.nms_iou_threshold
                    )
                    or (
                        self.containment_threshold is not None
                        and intersection_over_smaller_box(
                            detection.bbox, kept.bbox
                        )
                        >= self.containment_threshold
                    )
                    or self._is_likely_person_duplicate(detection, kept)
                )
                for kept in result
            ):
                continue
            result.append(detection)
        return result

    def _is_likely_person_duplicate(
        self, detection: Detection, kept: Detection
    ) -> bool:
        threshold = self.person_duplicate_containment_threshold
        distance_ratio = self.person_duplicate_footpoint_distance_ratio
        center_distance_ratio = self.person_duplicate_center_distance_ratio
        min_height_ratio = self.person_duplicate_min_height_ratio
        if (
            threshold is None
            or distance_ratio is None
            or center_distance_ratio is None
            or min_height_ratio is None
            or detection.class_name not in PERSON_CLASSES
            or kept.class_name != detection.class_name
            or intersection_over_smaller_box(detection.bbox, kept.bbox) < threshold
        ):
            return False
        if (
            detection.metadata.get("team_id") is not None
            and kept.metadata.get("team_id") is not None
            and detection.metadata["team_id"] != kept.metadata["team_id"]
        ):
            return False
        larger_height = max(detection.bbox.height, kept.bbox.height)
        smaller_height = min(detection.bbox.height, kept.bbox.height)
        if larger_height <= 0 or smaller_height / larger_height < min_height_ratio:
            return False
        left_foot = detection.bbox.footpoint
        right_foot = kept.bbox.footpoint
        footpoint_distance = hypot(
            left_foot[0] - right_foot[0],
            left_foot[1] - right_foot[1],
        )
        left_center = detection.bbox.center
        right_center = kept.bbox.center
        center_distance = hypot(
            left_center[0] - right_center[0],
            left_center[1] - right_center[1],
        )
        if (
            footpoint_distance > larger_height * distance_ratio
            or center_distance > larger_height * center_distance_ratio
        ):
            return False
        appearance_distance = _appearance_distance(
            detection.appearance, kept.appearance
        )
        max_appearance_distance = self.person_duplicate_max_appearance_distance
        return (
            appearance_distance is None
            or max_appearance_distance is None
            or appearance_distance <= max_appearance_distance
        )


def _appearance_distance(
    appearance_a: tuple[float, ...] | None,
    appearance_b: tuple[float, ...] | None,
) -> float | None:
    if (
        appearance_a is None
        or appearance_b is None
        or len(appearance_a) != len(appearance_b)
        or not appearance_a
    ):
        return None
    dot = sum(left * right for left, right in zip(appearance_a, appearance_b))
    norm_a = sqrt(sum(value * value for value in appearance_a))
    norm_b = sqrt(sum(value * value for value in appearance_b))
    if norm_a == 0 or norm_b == 0:
        return None
    similarity = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return (1.0 - similarity) / 2.0


def bbox_from(value: BoundingBox | Sequence[float]) -> BoundingBox:
    """Create a box from an existing box or four-number sequence."""
    if isinstance(value, BoundingBox):
        return value
    if len(value) != 4:
        raise ValueError("A bounding box must contain four coordinates")
    return BoundingBox(*map(float, value))
