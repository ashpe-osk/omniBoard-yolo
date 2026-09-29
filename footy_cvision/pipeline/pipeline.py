"""Detector-agnostic frame pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from math import hypot
from typing import Any

from footy_cvision.camera.motion import estimate_motion
from footy_cvision.core import Detection, FrameResult, Track
from footy_cvision.detection.filter import DetectionFilter, bbox_from
from footy_cvision.football.ball_assignment import BallAssignment, assign_ball
from footy_cvision.football.ball_trajectory import (
    BallTrajectory,
    BallTrajectoryConfig,
)
from footy_cvision.football.constraints import (
    dead_ball_check,
    validate_temporal_positions,
)
from footy_cvision.football.pitch_geometry import PitchGeometry
from footy_cvision.football.possession import PossessionTracker
from footy_cvision.tracking.tracker import Tracker, TrackerConfig

Detector = Callable[[Any], Iterable[Detection | Mapping[str, Any]]]


@dataclass(frozen=True)
class PipelineConfig:
    tracker: TrackerConfig = TrackerConfig()
    min_confidence: float = 0.0
    ball_class: str = "ball"
    player_classes: tuple[str, ...] = ("player",)
    max_ball_distance: float = 120.0
    ball_min_confidence: float = 0.0
    pitch_polygon: tuple[tuple[float, float], ...] | None = None
    max_ball_distance_m: float = 3.0
    possession_confirm_observations: int = 2
    possession_release_observations: int = 2
    max_speed: float = 250.0
    nms_iou_threshold: float | None = 0.5
    containment_threshold: float | None = 0.90
    person_duplicate_containment_threshold: float | None = 0.75
    person_duplicate_footpoint_distance_ratio: float | None = 0.20
    person_duplicate_center_distance_ratio: float | None = 0.50
    person_duplicate_min_height_ratio: float | None = 0.55
    person_duplicate_max_appearance_distance: float | None = 0.15
    pitch_geometry: PitchGeometry | None = None
    ball_trajectory: BallTrajectoryConfig = field(
        default_factory=BallTrajectoryConfig
    )


class Pipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.config = config or PipelineConfig()
        self.tracker = Tracker(self.config.tracker)
        self.filter = DetectionFilter(
            min_confidence=self.config.min_confidence,
            pitch_polygon=self.config.pitch_polygon,
            filter_out_of_bounds=self.config.pitch_polygon is not None,
            nms_iou_threshold=self.config.nms_iou_threshold,
            containment_threshold=self.config.containment_threshold,
            person_duplicate_containment_threshold=(
                self.config.person_duplicate_containment_threshold
            ),
            person_duplicate_footpoint_distance_ratio=(
                self.config.person_duplicate_footpoint_distance_ratio
            ),
            person_duplicate_center_distance_ratio=(
                self.config.person_duplicate_center_distance_ratio
            ),
            person_duplicate_min_height_ratio=(
                self.config.person_duplicate_min_height_ratio
            ),
            person_duplicate_max_appearance_distance=(
                self.config.person_duplicate_max_appearance_distance
            ),
        )
        self._previous_image: Any = None
        self._previous_tracks: dict[int, Track] = {}
        self._previous_frame_index: int | None = None
        self._previous_ball_player: int | None = None
        self.possession = PossessionTracker(
            confirm_observations=self.config.possession_confirm_observations,
            release_observations=self.config.possession_release_observations,
        )
        self.ball_trajectory = (
            BallTrajectory(self.config.ball_trajectory, self.config.ball_class)
            if self.config.pitch_geometry is not None
            else None
        )

    @staticmethod
    def _coerce_detection(value: Detection | Mapping[str, Any]) -> Detection:
        if isinstance(value, Detection):
            return value
        return Detection(
            bbox=bbox_from(value["bbox"]),
            confidence=float(value["confidence"]),
            class_name=str(value["class_name"]),
            appearance=(
                tuple(map(float, value["appearance"]))
                if value.get("appearance") is not None
                else None
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def process_frame(self, frame: Any, frame_index: int) -> FrameResult:
        if self.detector is not None:
            raw_detections = self.detector(frame)
        elif isinstance(frame, Mapping) and "detections" in frame:
            raw_detections = frame["detections"]
        else:
            raise ValueError(
                "No detector supplied and frame does not contain detections"
            )
        coerced_detections = [
            self._coerce_detection(item) for item in raw_detections
        ]
        raw_ball_candidates = [
            detection
            for detection in coerced_detections
            if detection.class_name == self.config.ball_class
            and detection.confidence >= self.config.min_confidence
        ]
        detections = self.filter(coerced_detections)
        image = frame.get("image") if isinstance(frame, Mapping) else frame
        motion = estimate_motion(self._previous_image, image)
        tracks = self.tracker.update(
            detections,
            frame_index,
            camera_transform=motion.matrix if motion.valid else None,
        )
        motion_record = motion.to_dict()
        motion_record["association_compensated_track_count"] = (
            self.tracker.last_camera_compensated_track_count
        )
        motion_record["association_used"] = (
            self.tracker.last_camera_compensated_track_count > 0
        )
        homography = None
        geometry = self.config.pitch_geometry
        if geometry is not None:
            keypoints = (
                frame.get("pitch_keypoints", {})
                if isinstance(frame, Mapping)
                else {}
            )
            confidences = (
                frame.get("pitch_keypoint_confidences")
                if isinstance(frame, Mapping)
                else None
            )
            if not isinstance(keypoints, Mapping):
                raise ValueError("pitch_keypoints must be a mapping")
            if confidences is not None and not isinstance(confidences, Mapping):
                raise ValueError("pitch_keypoint_confidences must be a mapping")
            homography = geometry.estimate(keypoints, confidences)
            for track in tracks:
                pitch_position = geometry.image_to_pitch(track.bbox.footpoint)
                track.metadata["pitch_position"] = (
                    list(pitch_position) if pitch_position is not None else None
                )
                track.metadata["homography_status"] = homography.status
                track.metadata["homography_quality"] = homography.quality
        ball_candidates = [
            detection
            for detection in detections
            if detection.class_name == self.config.ball_class
        ]
        ball = (
            max(ball_candidates, key=lambda detection: detection.confidence)
            if ball_candidates
            else None
        )
        players = [
            track
            for track in tracks
            if track.class_name in self.config.player_classes
        ]
        assignment: BallAssignment = assign_ball(
            ball,
            players,
            previous_player_id=self._previous_ball_player,
            min_confidence=self.config.ball_min_confidence,
            max_distance=self.config.max_ball_distance,
            max_distance_m=self.config.max_ball_distance_m,
            ball_pitch_position=(
                geometry.image_to_pitch(ball.bbox.footpoint)
                if geometry is not None and ball is not None
                else None
            ),
        )
        if assignment.player_id is not None:
            self._previous_ball_player = assignment.player_id
        frame_delta = (
            max(1, frame_index - self._previous_frame_index)
            if self._previous_frame_index is not None
            else 1
        )
        violations = validate_temporal_positions(
            tracks,
            self._previous_tracks,
            frame_delta=frame_delta,
            max_speed=self.config.max_speed,
        )
        for track in tracks:
            previous = self._previous_tracks.get(track.track_id)
            if previous is not None:
                track.metadata["speed"] = hypot(
                    track.bbox.center[0] - previous.bbox.center[0],
                    track.bbox.center[1] - previous.bbox.center[1],
                ) / frame_delta
        trajectory_point = None
        if self.ball_trajectory is not None and geometry is not None:
            trajectory_point = self.ball_trajectory.process(
                ball_candidates, geometry, frame_index
            )
            if trajectory_point["reason"] == "pitch_space_outlier":
                violations.append("ball:pitch_space_outlier")
        dead_ball = (
            max(raw_ball_candidates, key=lambda item: item.confidence)
            if raw_ball_candidates
            else None
        )
        if dead_ball_check(dead_ball, self.config.pitch_polygon):
            violations.append("ball:outside_pitch")
        possession_state, events = self.possession.update(
            frame_index=frame_index,
            candidate_player_id=assignment.player_id,
            candidate_confidence=assignment.confidence,
            tracks=tracks,
        )
        self._previous_image = image
        self._previous_tracks = {
            track.track_id: replace(track, metadata=dict(track.metadata))
            for track in tracks
        }
        self._previous_frame_index = frame_index
        return FrameResult(
            frame_index=frame_index,
            detections=detections,
            tracks=self.tracker.snapshot(include_lost=True),
            ball_assignment=assignment.to_dict(),
            camera_motion=motion_record,
            violations=violations,
            homography=homography.to_dict() if homography is not None else None,
            ball_trajectory=trajectory_point,
            possession=possession_state,
            events=events,
        )

    def run(self, frames: Iterable[Any]) -> list[FrameResult]:
        return [self.process_frame(frame, index) for index, frame in enumerate(frames)]


def records_to_jsonable(results: Iterable[FrameResult]) -> list[dict[str, Any]]:
    return [result.to_dict() for result in results]
