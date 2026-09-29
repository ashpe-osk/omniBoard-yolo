"""Optional OpenCV-based affine/translation estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MotionEstimate:
    matrix: tuple[tuple[float, float, float], ...] | None
    method: str
    valid: bool
    reason: str | None = None
    inlier_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_motion(previous_frame: Any, current_frame: Any) -> MotionEstimate:
    """Estimate previous-to-current frame motion; return an explicit no-op if unavailable."""
    if previous_frame is None or current_frame is None:
        return MotionEstimate(None, "none", False, "missing_frame")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return MotionEstimate(None, "none", False, "opencv_unavailable")

    try:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray, maxCorners=500, qualityLevel=0.01, minDistance=8
        )
        if previous_points is None or len(previous_points) < 3:
            return MotionEstimate(None, "none", False, "insufficient_features")
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, previous_points, None
        )
        if current_points is None or status is None:
            return MotionEstimate(None, "none", False, "optical_flow_failed")
        valid = status.reshape(-1) == 1
        source = previous_points.reshape(-1, 2)[valid]
        target = current_points.reshape(-1, 2)[valid]
        if len(source) < 3:
            return MotionEstimate(None, "none", False, "insufficient_matches")
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is None:
            return MotionEstimate(None, "none", False, "affine_estimation_failed")
        normalized = tuple(
            tuple(float(value) for value in row) for row in matrix.tolist()
        )
        inlier_count = int(inliers.sum()) if inliers is not None else len(source)
        return MotionEstimate(normalized, "affine", True, None, inlier_count)
    except (AttributeError, TypeError, ValueError, cv2.error) as error:
        return MotionEstimate(None, "none", False, f"opencv_error:{error}")
