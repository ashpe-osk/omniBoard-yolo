"""Jersey color features and a deterministic two-cluster team classifier."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Sequence

from footy_cvision.core import BoundingBox

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
TEAM_IDS = (TEAM_A, TEAM_B)


def extract_color_feature(
    frame: Any, bbox: BoundingBox, vertical_fraction: float = 0.65
) -> tuple[float, ...] | None:
    """Return mean RGB color from the upper player crop, supporting ndarray/list images."""
    if not 0 < vertical_fraction <= 1:
        raise ValueError("vertical_fraction must be in (0, 1]")
    try:
        height, width = int(frame.shape[0]), int(frame.shape[1])
        channels = int(frame.shape[2]) if len(frame.shape) > 2 else 1
        get_pixel = lambda y, x: frame[y, x] if hasattr(frame, "shape") else frame[y][x]
    except (AttributeError, IndexError, TypeError):
        try:
            height, width = len(frame), len(frame[0])
            channels = len(frame[0][0]) if isinstance(frame[0][0], (list, tuple)) else 1
            get_pixel = lambda y, x: frame[y][x]
        except (IndexError, TypeError):
            return None
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(y1, min(height, int(y1 + (bbox.y2 - bbox.y1) * vertical_fraction)))
    if x2 <= x1 or y2 <= y1:
        return None
    sums = [0.0] * min(channels, 3)
    count = 0
    stride = max(1, ((x2 - x1) * (y2 - y1)) // 10000)
    for y in range(y1, y2):
        for x in range(x1, x2):
            if (y * width + x) % stride:
                continue
            pixel = get_pixel(y, x)
            values = [float(pixel)] if channels == 1 else [float(pixel[i]) for i in range(3)]
            for index, value in enumerate(values):
                sums[index] += value
            count += 1
    if count == 0:
        return None
    return tuple(value / count for value in sums)


def extract_team_color_feature(
    frame: Any, bbox: BoundingBox
) -> tuple[float, ...] | None:
    """Return the dominant BGR color from a green-filtered torso crop."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "OpenCV and NumPy are required for jersey-color features"
        ) from error
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(y1, min(height, int(bbox.y2)))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 12 or box_height < 32:
        return None
    player_crop = frame[y1:y2, x1:x2]
    crop = player_crop[
        int(box_height * 0.20):int(box_height * 0.65),
        int(box_width * 0.20):int(box_width * 0.80),
    ]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 30), (95, 255, 255))
    pixels = crop[green_mask == 0].astype(np.float32)
    if len(pixels) < 20:
        pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 2:
        return None
    if len(pixels) > 2000:
        sample_indices = np.linspace(0, len(pixels) - 1, 2000, dtype=int)
        pixels = pixels[sample_indices]

    centers = np.stack((pixels[0], pixels[0]))
    distances = np.sum((pixels - centers[0]) ** 2, axis=1)
    centers[1] = pixels[int(np.argmax(distances))]
    labels = np.zeros(len(pixels), dtype=np.int8)
    for _ in range(10):
        distances = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            break
        updated = np.stack(
            [pixels[labels == index].mean(axis=0) for index in range(2)]
        )
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    counts = np.bincount(labels, minlength=2)
    dominant = int(np.argmax(counts))
    return tuple(float(channel / 255.0) for channel in centers[dominant])


class TeamClassifier:
    """Two-kit appearance model with stable semantic labels and confidence.

    The dominant torso colors are clustered independent of detection order.
    BGR feature ordering gives deterministic semantic labels; for the sample
    red/white kits, red is TEAM_A and white is TEAM_B.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        minimum_cluster_separation: float = 0.08,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if minimum_cluster_separation < 0:
            raise ValueError("minimum_cluster_separation cannot be negative")
        self.max_iterations = max_iterations
        self.minimum_cluster_separation = minimum_cluster_separation
        self.centers: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def fit(self, features: Sequence[Sequence[float]]) -> TeamClassifier:
        points = [tuple(map(float, feature)) for feature in features if feature]
        if len(points) < 2:
            raise ValueError("At least two non-empty color features are required")
        dimensions = len(points[0])
        if any(len(point) != dimensions for point in points):
            raise ValueError("All color features must have equal dimensions")
        first = points[0]
        second = max(points[1:], key=lambda point: self._distance(first, point))
        centers = [first, second]
        for _ in range(self.max_iterations):
            groups: list[list[tuple[float, ...]]] = [[], []]
            for point in points:
                cluster = min(
                    range(2),
                    key=lambda index: (self._distance(point, centers[index]), index),
                )
                groups[cluster].append(point)
            updated = [
                tuple(
                    sum(point[i] for point in group) / len(group)
                    for i in range(dimensions)
                )
                if group
                else centers[index]
                for index, group in enumerate(groups)
            ]
            if updated == centers:
                break
            centers = updated
        if self._distance(centers[0], centers[1]) < self.minimum_cluster_separation:
            raise ValueError("Player appearance samples do not separate two kits")
        ordered = sorted(centers, key=self._semantic_key)
        self.centers = (ordered[0], ordered[1])
        return self

    def predict(self, feature: Sequence[float]) -> int:
        """Return the semantic team index: 0 for TEAM_A, 1 for TEAM_B."""
        team_index, _ = self.predict_with_confidence(feature)
        return team_index

    def predict_with_confidence(
        self, feature: Sequence[float]
    ) -> tuple[int, float]:
        """Return semantic team index and a bounded centroid-separation score."""
        if self.centers is None:
            raise RuntimeError("TeamClassifier must be fitted before prediction")
        point = tuple(map(float, feature))
        if len(point) != len(self.centers[0]):
            raise ValueError("Feature dimensions do not match fitted centers")
        distances = [self._distance(point, center) for center in self.centers]
        winner = min(range(2), key=lambda index: (distances[index], index))
        best, other = sorted(distances)
        confidence = (other - best) / max(other, 1e-9)
        return winner, max(0.0, min(1.0, confidence))

    @staticmethod
    def _semantic_key(feature: Sequence[float]) -> tuple[float, ...]:
        if len(feature) not in {18 * 8, 18 * 8 + 8}:
            return tuple(round(value, 8) for value in feature)
        histogram = [
            [float(feature[hue * 8 + saturation]) for saturation in range(8)]
            for hue in range(18)
        ]
        total = sum(sum(row) for row in histogram)
        if total <= 1e-12:
            return (0.0, 0.0, *tuple(round(value, 8) for value in feature))
        saturation_values = [((index + 0.5) / 8) for index in range(8)]
        saturation = sum(
            histogram[hue][sat] * saturation_values[sat]
            for hue in range(18)
            for sat in range(8)
        ) / total
        hue_weights = [
            sum(histogram[hue][sat] * saturation_values[sat] for sat in range(8))
            for hue in range(18)
        ]
        hue_total = sum(hue_weights)
        if hue_total:
            hue = sum((index + 0.5) / 18 * weight for index, weight in enumerate(hue_weights)) / hue_total
        else:
            hue = 1.0
        value = (
            sum(
                float(feature[18 * 8 + index]) * ((index + 0.5) / 8)
                for index in range(8)
            )
            / sum(float(item) for item in feature[18 * 8:])
            if len(feature) == 18 * 8 + 8
            and sum(float(item) for item in feature[18 * 8:]) > 1e-12
            else 0.0
        )
        return (
            -saturation,
            hue,
            -value,
            *tuple(round(item, 8) for item in feature),
        )

    @staticmethod
    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


@dataclass
class _TrackTeamVotes:
    observations_by_team: Counter[str] = field(default_factory=Counter)
    recent_votes: deque[tuple[str, float]] = field(default_factory=deque)
    observations: int = 0
    assigned_team: str | None = None
    history: list[str] = field(default_factory=list)


class TeamStabilizer:
    """Track-level confidence-weighted voting with a hysteresis switch gate."""

    def __init__(
        self,
        minimum_switch_observations: int = 3,
        switch_weight_ratio: float = 1.5,
        stable_observations: int = 3,
        minimum_confidence: float = 0.60,
        vote_window_size: int = 10,
    ) -> None:
        if (
            minimum_switch_observations < 1
            or stable_observations < 1
            or vote_window_size < 1
        ):
            raise ValueError("Team observation thresholds must be positive")
        if switch_weight_ratio <= 1 or not 0 <= minimum_confidence <= 1:
            raise ValueError("Invalid team stabilization thresholds")
        self.minimum_switch_observations = minimum_switch_observations
        self.switch_weight_ratio = switch_weight_ratio
        self.stable_observations = stable_observations
        self.minimum_confidence = minimum_confidence
        self.vote_window_size = vote_window_size
        self._tracks: dict[int, _TrackTeamVotes] = {}

    def update(
        self, track_id: int, team_id: str, confidence: float
    ) -> dict[str, Any]:
        if team_id not in TEAM_IDS:
            raise ValueError(f"Unknown team identity: {team_id}")
        if not 0 <= confidence <= 1:
            raise ValueError("Team confidence must be between 0 and 1")
        state = self._tracks.setdefault(
            track_id,
            _TrackTeamVotes(recent_votes=deque(maxlen=self.vote_window_size)),
        )
        state.observations += 1
        state.observations_by_team[team_id] += 1
        state.recent_votes.append((team_id, confidence))
        weights, recent_counts, recent_confidence = self._recent_statistics(state)
        winner = max(TEAM_IDS, key=lambda team: (weights[team], team == TEAM_A))
        if state.assigned_team is None:
            state.assigned_team = winner
            state.history.append(winner)
        elif winner != state.assigned_team:
            old_weight = weights[state.assigned_team]
            if (
                recent_counts[winner]
                >= self.minimum_switch_observations
                and weights[winner]
                >= old_weight * self.switch_weight_ratio
            ):
                state.assigned_team = winner
                state.history.append(winner)
        total_weight = sum(weights.values())
        team_confidence = (
            weights[state.assigned_team] / total_weight * recent_confidence
            if state.assigned_team is not None and total_weight
            else 0.0
        )
        opposing_streak = 0
        for recent_team, _ in reversed(state.recent_votes):
            if recent_team == state.assigned_team:
                break
            opposing_streak += 1
        uncertain = (
            state.observations < self.stable_observations
            or team_confidence < self.minimum_confidence
            or opposing_streak >= 2
        )
        return {
            "team_id": state.assigned_team,
            "team_confidence": max(0.0, min(1.0, team_confidence)),
            "team_uncertain": uncertain,
            "team_history": list(state.history),
            "team_observations": state.observations,
            "observations_by_team": dict(state.observations_by_team),
        }

    def report(self) -> dict[str, Any]:
        return {
            str(track_id): {
                "team_id": state.assigned_team,
                "team_history": list(state.history),
                "team_observations": state.observations,
                "team_confidence": self._track_confidence(state),
            }
            for track_id, state in self._tracks.items()
        }

    @staticmethod
    def _recent_statistics(
        state: _TrackTeamVotes,
    ) -> tuple[Counter[str], Counter[str], float]:
        weights: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        confidence_sum = 0.0
        for team_id, confidence in state.recent_votes:
            weights[team_id] += max(confidence, 0.05)
            counts[team_id] += 1
            confidence_sum += confidence
        mean_confidence = (
            confidence_sum / len(state.recent_votes) if state.recent_votes else 0.0
        )
        return weights, counts, mean_confidence

    @classmethod
    def _track_confidence(cls, state: _TrackTeamVotes) -> float:
        weights, _, recent_confidence = cls._recent_statistics(state)
        total_weight = sum(weights.values())
        if state.assigned_team is None or not total_weight:
            return 0.0
        return weights[state.assigned_team] / total_weight * recent_confidence


class TeamFeatureHistory:
    """Smooth per-track jersey features before assigning a team cluster."""

    def __init__(self, window_size: int = 10) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self._features: dict[int, deque[tuple[float, ...]]] = {}

    def update(
        self, track_id: int, feature: Sequence[float]
    ) -> tuple[float, ...]:
        values = tuple(map(float, feature))
        if not values:
            raise ValueError("Team features must not be empty")
        history = self._features.setdefault(
            track_id,
            deque(maxlen=self.window_size),
        )
        if history and len(values) != len(history[0]):
            raise ValueError("Team feature dimensions changed for a track")
        history.append(values)
        mean = tuple(
            sum(sample[index] for sample in history) / len(history)
            for index in range(len(values))
        )
        return mean

    def reset(self, track_id: int, feature: Sequence[float]) -> tuple[float, ...]:
        values = tuple(map(float, feature))
        if not values:
            raise ValueError("Team features must not be empty")
        history = self._features.get(track_id)
        if history and len(values) != len(history[0]):
            raise ValueError("Team feature dimensions changed for a track")
        self._features[track_id] = deque([values], maxlen=self.window_size)
        return values
