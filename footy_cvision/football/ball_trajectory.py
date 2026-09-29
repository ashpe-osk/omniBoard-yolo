"""Ball selection and pitch-space trajectory validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any, TYPE_CHECKING

from footy_cvision.core import Detection

if TYPE_CHECKING:
    from footy_cvision.football.pitch_geometry import PitchGeometry


def select_highest_confidence_ball(
    detections: Iterable[Detection],
    ball_class: str = "ball",
    min_confidence: float = 0.0,
) -> Detection | None:
    """Return the highest-confidence eligible ball detection, if any."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    candidates = (
        detection
        for detection in detections
        if detection.class_name == ball_class
        and detection.confidence >= min_confidence
    )
    return max(candidates, key=lambda detection: detection.confidence, default=None)


@dataclass(frozen=True)
class BallTrajectoryConfig:
    max_speed_m_per_frame: float = 12.0
    min_confidence: float = 0.0
    max_frame_gap: int = 10

    def __post_init__(self) -> None:
        if (
            not isfinite(self.max_speed_m_per_frame)
            or self.max_speed_m_per_frame <= 0
        ):
            raise ValueError("max_speed_m_per_frame must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.max_frame_gap < 1:
            raise ValueError("max_frame_gap must be positive")


class BallTrajectory:
    """Produce one interpolation-friendly point record for each processed frame."""

    def __init__(
        self,
        config: BallTrajectoryConfig | None = None,
        ball_class: str = "ball",
    ) -> None:
        self.config = config or BallTrajectoryConfig()
        self.ball_class = ball_class
        self.points: list[dict[str, Any]] = []
        self._last_valid_position: tuple[float, float] | None = None
        self._last_valid_frame: int | None = None

    def process(
        self,
        detections: Iterable[Detection],
        geometry: PitchGeometry,
        frame_index: int,
    ) -> dict[str, Any]:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.points and frame_index <= self.points[-1]["frame_index"]:
            raise ValueError("frame_index must increase monotonically")

        ball = select_highest_confidence_ball(detections, self.ball_class)
        result: dict[str, Any] = {
            "frame_index": frame_index,
            "confidence": ball.confidence if ball is not None else None,
            "image_position": list(ball.bbox.center) if ball is not None else None,
            "raw_pitch_position": None,
            "pitch_position": None,
            "valid": False,
            "interpolated": False,
            "reason": None,
        }
        if ball is None:
            result["reason"] = "ball_not_detected"
        elif ball.confidence < self.config.min_confidence:
            result["reason"] = "low_confidence"
        else:
            projected = geometry.image_to_pitch(ball.bbox.center)
            if projected is None:
                result["reason"] = "homography_unavailable"
            else:
                result["raw_pitch_position"] = list(projected)
                elapsed = (
                    frame_index - self._last_valid_frame
                    if self._last_valid_frame is not None
                    else None
                )
                if elapsed is not None and elapsed <= self.config.max_frame_gap:
                    assert self._last_valid_position is not None
                    speed = hypot(
                        projected[0] - self._last_valid_position[0],
                        projected[1] - self._last_valid_position[1],
                    ) / elapsed
                    result["speed_m_per_frame"] = speed
                    if speed > self.config.max_speed_m_per_frame:
                        result["reason"] = "pitch_space_outlier"
                    else:
                        result["pitch_position"] = list(projected)
                        result["valid"] = True
                else:
                    result["pitch_position"] = list(projected)
                    result["valid"] = True
                if result["valid"]:
                    self._last_valid_position = projected
                    self._last_valid_frame = frame_index

        self.points.append(result)
        return result

    def reset(self) -> None:
        self.points.clear()
        self._last_valid_position = None
        self._last_valid_frame = None
