"""Distance-based ball-to-player assignment with confidence and hysteresis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Sequence

from footy_cvision.core import Detection, Track


@dataclass(frozen=True)
class BallAssignment:
    player_id: int | None
    distance: float | None
    confidence: float
    reason: str | None = None
    coordinate_space: str = "image_px"

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "ball_nearest_player": self.player_id,
            "ball_distance_to_player": self.distance,
            "ball_assignment_confidence": self.confidence,
            "coordinate_space": self.coordinate_space,
        }


def assign_ball(
    ball: Detection | None,
    players: Sequence[Track],
    previous_player_id: int | None = None,
    min_confidence: float = 0.0,
    max_distance: float = 120.0,
    hysteresis: float = 0.15,
    ball_pitch_position: tuple[float, float] | None = None,
    max_distance_m: float = 3.0,
) -> BallAssignment:
    if (
        not 0 <= min_confidence <= 1
        or max_distance <= 0
        or max_distance_m <= 0
        or hysteresis < 0
    ):
        raise ValueError("Invalid ball-assignment thresholds")
    if ball is None:
        return BallAssignment(None, None, 0.0, "ball_not_detected")
    if ball.confidence < min_confidence:
        return BallAssignment(None, None, ball.confidence, "low_confidence")
    ball_x, ball_y = ball.bbox.center
    use_pitch = ball_pitch_position is not None and all(
        player.metadata.get("pitch_position") is not None for player in players
    )
    if use_pitch:
        ball_x, ball_y = ball_pitch_position
    coordinate_space = "pitch_m" if use_pitch else "image_px"
    candidates: list[tuple[float, int]] = []
    for player in players:
        if use_pitch:
            position = player.metadata["pitch_position"]
            x, y = float(position[0]), float(position[1])
        else:
            x, y = player.bbox.footpoint
        candidates.append((hypot(ball_x - x, ball_y - y), player.track_id))
    if not candidates:
        return BallAssignment(None, None, ball.confidence, "no_players")
    candidates.sort(key=lambda item: (item[0], item[1]))
    best_distance, best_id = candidates[0]
    previous = next(
        (candidate for candidate in candidates if candidate[1] == previous_player_id),
        None,
    )
    if previous is not None and previous[0] <= best_distance * (1.0 + hysteresis):
        best_distance, best_id = previous
    reach = max_distance_m if use_pitch else max_distance
    if best_distance > reach:
        return BallAssignment(
            None, best_distance, 0.0, "out_of_reach", coordinate_space
        )
    proximity_confidence = max(0.0, 1.0 - best_distance / reach)
    confidence = ball.confidence * proximity_confidence
    return BallAssignment(
        best_id,
        best_distance,
        confidence,
        coordinate_space=coordinate_space,
    )
