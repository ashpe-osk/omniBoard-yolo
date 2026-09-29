"""Temporal validation and basic dead-ball boundary checks."""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot

from footy_cvision.core import Detection, Track
from footy_cvision.detection.filter import Point, point_in_polygon


def dead_ball_check(
    ball: Detection | None, pitch_polygon: Sequence[Point] | None = None
) -> bool:
    """A detected ball is considered dead when its center is outside the pitch."""
    if ball is None or pitch_polygon is None:
        return False
    return not point_in_polygon(ball.bbox.center, pitch_polygon)


def validate_temporal_positions(
    current: Sequence[Track],
    previous_by_id: dict[int, Track],
    frame_delta: int = 1,
    max_speed: float = 250.0,
) -> list[str]:
    """Report implausible observed displacements; never mutate the input tracks."""
    if frame_delta < 1 or max_speed <= 0:
        raise ValueError("frame_delta and max_speed must be positive")
    violations: list[str] = []
    for track in current:
        previous = previous_by_id.get(track.track_id)
        if previous is None:
            continue
        speed = hypot(
            track.bbox.center[0] - previous.bbox.center[0],
            track.bbox.center[1] - previous.bbox.center[1],
        ) / frame_delta
        if speed > max_speed:
            violations.append(f"track:{track.track_id}:speed:{speed:.2f}")
    return violations
