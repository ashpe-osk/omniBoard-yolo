"""Optional image-to-pitch calibration and coordinate transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any

Point = tuple[float, float]
Matrix3 = tuple[tuple[float, float, float], ...]


def _standard_landmarks(length_m: float, width_m: float) -> dict[str, Point]:
    return {
        "top_left": (0.0, 0.0),
        "top_right": (length_m, 0.0),
        "bottom_right": (length_m, width_m),
        "bottom_left": (0.0, width_m),
        "center": (length_m / 2, width_m / 2),
    }


@dataclass(frozen=True)
class SoccerPitchConfig:
    """Metric pitch dimensions and named keypoint destinations."""

    length_m: float = 105.0
    width_m: float = 68.0
    landmarks: Mapping[str, Point] | None = None
    min_keypoints: int = 4
    min_keypoint_confidence: float = 0.25
    max_reprojection_error_m: float = 3.0
    ransac_threshold_m: float = 1.5
    smoothing_alpha: float = 0.65
    max_reuse_frames: int = 3

    def __post_init__(self) -> None:
        if (
            not isfinite(self.length_m)
            or not isfinite(self.width_m)
            or self.length_m <= 0
            or self.width_m <= 0
        ):
            raise ValueError("Pitch dimensions must be positive")
        if self.min_keypoints < 4:
            raise ValueError("At least four keypoints are required")
        if not 0 <= self.min_keypoint_confidence <= 1:
            raise ValueError("min_keypoint_confidence must be between 0 and 1")
        if (
            not isfinite(self.max_reprojection_error_m)
            or not isfinite(self.ransac_threshold_m)
            or self.max_reprojection_error_m <= 0
            or self.ransac_threshold_m <= 0
        ):
            raise ValueError("Homography error thresholds must be positive")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.max_reuse_frames < 0:
            raise ValueError("max_reuse_frames must be non-negative")
        if self.landmarks is None:
            object.__setattr__(
                self, "landmarks", _standard_landmarks(self.length_m, self.width_m)
            )
        if not self.landmarks:
            raise ValueError("At least one named pitch landmark is required")
        for name, point in self.landmark_vertices.items():
            if not name or len(point) != 2 or not all(isfinite(v) for v in point):
                raise ValueError("Pitch landmarks must be named finite 2D points")

    @property
    def landmark_vertices(self) -> Mapping[str, Point]:
        assert self.landmarks is not None
        return self.landmarks


@dataclass(frozen=True)
class HomographyResult:
    matrix: Matrix3 | None
    status: str
    valid: bool
    reused: bool
    keypoint_count: int
    inlier_count: int
    mean_reprojection_error_m: float | None
    quality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "valid": self.valid,
            "reused": self.reused,
            "keypoint_count": self.keypoint_count,
            "inlier_count": self.inlier_count,
            "mean_reprojection_error_m": self.mean_reprojection_error_m,
            "quality": self.quality,
        }


class PitchGeometry:
    """Estimate and reuse a homography from named image landmarks.

    ``image_keypoints`` maps landmark names to ``(x, y)`` image pixels or
    ``(x, y, confidence)``. A separate confidence mapping can also be supplied.
    OpenCV is used for RANSAC when installed; otherwise normalized DLT via
    NumPy is used. Neither dependency is imported until estimation is called.
    """

    def __init__(self, config: SoccerPitchConfig | None = None) -> None:
        self.config = config or SoccerPitchConfig()
        self._last_valid_matrix: Matrix3 | None = None
        self._last_result: HomographyResult | None = None
        self._failed_frames = 0

    @property
    def result(self) -> HomographyResult:
        return self._last_result or HomographyResult(
            None, "not_estimated", False, False, 0, 0, None, 0.0
        )

    def estimate(
        self,
        image_keypoints: Mapping[str, Sequence[float]],
        confidences: Mapping[str, float] | None = None,
    ) -> HomographyResult:
        confidences = confidences or {}
        source: list[Point] = []
        destination: list[Point] = []
        accepted_confidences: list[float] = []
        for name, pitch_point in self.config.landmark_vertices.items():
            raw = image_keypoints.get(name)
            if raw is None or len(raw) < 2:
                continue
            confidence = float(
                confidences.get(name, raw[2] if len(raw) >= 3 else 1.0)
            )
            image_point = (float(raw[0]), float(raw[1]))
            if (
                not all(isfinite(value) for value in image_point)
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
                or confidence < self.config.min_keypoint_confidence
            ):
                continue
            source.append(image_point)
            destination.append((float(pitch_point[0]), float(pitch_point[1])))
            accepted_confidences.append(confidence)

        count = len(source)
        if count < self.config.min_keypoints:
            return self._failed("insufficient_keypoints", count)

        matrix, inlier_mask = self._fit_homography(source, destination)
        if matrix is None:
            return self._failed("degenerate_keypoints", count)

        inliers = [
            index
            for index, is_inlier in enumerate(inlier_mask)
            if is_inlier
        ]
        errors = self._reprojection_errors(
            matrix,
            [source[index] for index in inliers],
            [destination[index] for index in inliers],
        )
        mean_error = sum(errors) / len(errors) if errors else float("inf")
        mean_confidence = sum(accepted_confidences) / count
        inlier_ratio = len(inliers) / count
        quality = mean_confidence * inlier_ratio * max(
            0.0,
            1.0 - mean_error / self.config.max_reprojection_error_m,
        )
        if (
            len(inliers) < self.config.min_keypoints
            or not isfinite(mean_error)
            or mean_error > self.config.max_reprojection_error_m
        ):
            return self._failed(
                "poor_quality", count, len(inliers), mean_error, quality
            )

        smoothed = self._smooth(matrix)
        if smoothed is not None:
            smoothed_errors = self._reprojection_errors(
                smoothed,
                [source[index] for index in inliers],
                [destination[index] for index in inliers],
            )
            if (
                smoothed_errors
                and sum(smoothed_errors) / len(smoothed_errors)
                <= self.config.max_reprojection_error_m
            ):
                matrix = smoothed

        errors = self._reprojection_errors(
            matrix,
            [source[index] for index in inliers],
            [destination[index] for index in inliers],
        )
        mean_error = sum(errors) / len(errors) if errors else float("inf")
        quality = mean_confidence * inlier_ratio * max(
            0.0,
            1.0 - mean_error / self.config.max_reprojection_error_m,
        )
        if (
            not isfinite(mean_error)
            or mean_error > self.config.max_reprojection_error_m
        ):
            return self._failed(
                "poor_quality", count, len(inliers), mean_error, quality
            )

        self._last_valid_matrix = matrix
        self._failed_frames = 0
        self._last_result = HomographyResult(
            matrix=_matrix_tuple(matrix),
            status="estimated",
            valid=True,
            reused=False,
            keypoint_count=count,
            inlier_count=len(inliers),
            mean_reprojection_error_m=mean_error,
            quality=quality,
        )
        return self._last_result

    def image_to_pitch(
        self, point: Sequence[float], matrix: Matrix3 | None = None
    ) -> Point | None:
        return _transform(point, matrix or self.result.matrix)

    def pitch_to_image(
        self, point: Sequence[float], matrix: Matrix3 | None = None
    ) -> Point | None:
        selected = matrix or self.result.matrix
        if selected is None:
            return None
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "NumPy is required for pitch coordinate transforms"
            ) from error
        inverse = np.linalg.inv(np.asarray(selected, dtype=float))
        return _transform(point, _matrix_tuple(inverse))

    def _failed(
        self,
        status: str,
        keypoint_count: int,
        inlier_count: int = 0,
        error: float | None = None,
        quality: float = 0.0,
    ) -> HomographyResult:
        self._failed_frames += 1
        if (
            self._last_valid_matrix is not None
            and self._failed_frames <= self.config.max_reuse_frames
        ):
            result = HomographyResult(
                matrix=self._last_valid_matrix,
                status="reused",
                valid=False,
                reused=True,
                keypoint_count=keypoint_count,
                inlier_count=inlier_count,
                mean_reprojection_error_m=error,
                quality=self._last_result.quality if self._last_result else 0.0,
            )
        else:
            result = HomographyResult(
                None,
                status,
                False,
                False,
                keypoint_count,
                inlier_count,
                error,
                quality,
            )
        self._last_result = result
        return result

    def _fit_homography(
        self, source: list[Point], destination: list[Point]
    ) -> tuple[Any | None, list[bool]]:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "Pitch homography estimation requires NumPy; install numpy"
            ) from error

        src = np.asarray(source, dtype=float)
        dst = np.asarray(destination, dtype=float)
        try:
            import cv2
        except ImportError:
            cv2 = None

        if cv2 is not None:
            method = cv2.RANSAC if len(source) > 4 else 0
            try:
                candidate, mask = cv2.findHomography(
                    src,
                    dst,
                    method,
                    self.config.ransac_threshold_m,
                )
            except cv2.error:
                candidate, mask = None, None
            if candidate is not None:
                candidate = _normalize_matrix(np.asarray(candidate, dtype=float))
                if candidate is not None:
                    inliers = (
                        np.asarray(mask).reshape(-1).astype(bool).tolist()
                        if mask is not None
                        else [True] * len(source)
                    )
                    return candidate, inliers

        try:
            source_normalized, source_transform = _normalize_points(src, np)
            dest_normalized, dest_transform = _normalize_points(dst, np)
        except ValueError:
            return None, []
        rows: list[list[float]] = []
        for (x, y), (u, v) in zip(source_normalized, dest_normalized):
            rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        try:
            _, singular_values, right = np.linalg.svd(
                np.asarray(rows, dtype=float), full_matrices=True
            )
        except np.linalg.LinAlgError:
            return None, []
        rank_check = singular_values[-2] if len(source) > 4 else singular_values[-1]
        if len(singular_values) < 8 or rank_check <= 1e-12:
            return None, []
        normalized = right[-1].reshape(3, 3)
        try:
            candidate = np.linalg.inv(dest_transform) @ normalized @ source_transform
        except np.linalg.LinAlgError:
            return None, []
        candidate = _normalize_matrix(candidate)
        if candidate is None or not np.isfinite(candidate).all():
            return None, []
        return candidate, [True] * len(source)

    def _smooth(self, matrix: Any) -> Any | None:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("NumPy is required for homography smoothing") from error
        if self._last_valid_matrix is None or self.config.smoothing_alpha == 1:
            return matrix
        alpha = self.config.smoothing_alpha
        prior = np.asarray(self._last_valid_matrix, dtype=float)
        return _normalize_matrix(alpha * matrix + (1 - alpha) * prior)

    @staticmethod
    def _reprojection_errors(
        matrix: Any, source: list[Point], destination: list[Point]
    ) -> list[float]:
        if not source:
            return []
        result: list[float] = []
        matrix_tuple = _matrix_tuple(matrix)
        for point, expected in zip(source, destination):
            actual = _transform(point, matrix_tuple)
            if actual is None:
                return [float("inf")]
            result.append(
                sqrt(
                    (actual[0] - expected[0]) ** 2
                    + (actual[1] - expected[1]) ** 2
                )
            )
        return result


def _normalize_points(points: Any, np: Any) -> tuple[Any, Any]:
    center = points.mean(axis=0)
    distances = np.sqrt(((points - center) ** 2).sum(axis=1))
    mean_distance = float(distances.mean())
    if mean_distance <= 1e-12:
        raise ValueError("Keypoints are degenerate")
    scale = sqrt(2.0) / mean_distance
    transform = np.asarray(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def _normalize_matrix(matrix: Any) -> Any | None:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for homography estimation") from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    scale = matrix[2, 2]
    if abs(scale) <= 1e-12:
        scale = float(np.linalg.norm(matrix))
    if abs(scale) <= 1e-12:
        return None
    matrix = matrix / scale
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        return None
    return matrix


def _matrix_tuple(matrix: Any) -> Matrix3:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _transform(point: Sequence[float], matrix: Matrix3 | None) -> Point | None:
    if matrix is None or len(point) != 2:
        return None
    x, y = float(point[0]), float(point[1])
    if not isfinite(x) or not isfinite(y):
        return None
    values = (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2],
    )
    if abs(values[2]) <= 1e-12:
        return None
    transformed = (values[0] / values[2], values[1] / values[2])
    return transformed if all(isfinite(value) for value in transformed) else None
