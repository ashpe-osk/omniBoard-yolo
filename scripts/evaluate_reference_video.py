"""Run a complete local-video detection, tracking, and annotation benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from footy_cvision.core import Detection, Track
from footy_cvision.detection.filter import DetectionFilter
from footy_cvision.detection.fusion import ModelAwareFusion, canonical_class_name
from footy_cvision.football.team_classification import (
    TEAM_A,
    TEAM_B,
    TeamClassifier,
    TeamFeatureHistory,
    TeamStabilizer,
    extract_team_color_feature,
)
from footy_cvision.models.backends import UltralyticsBackend
from footy_cvision.pipeline.pipeline import Pipeline, PipelineConfig
from footy_cvision.tracking.association import AssociationConfig
from footy_cvision.tracking.tracker import TrackerConfig
from footy_cvision.visualization.annotations import (
    draw_debug_frame,
    draw_tracking_frame,
)


PERSON_CLASSES = {"player", "goalkeeper", "referee"}
COLORS = {
    "player": (40, 220, 40),
    "goalkeeper": (0, 165, 255),
    "referee": (220, 80, 220),
    "ball": (0, 220, 255),
}
TEAM_FEATURE_RESET_CONFIDENCE = 0.75


def _feature(frame: Any, detection: Detection) -> tuple[float, ...] | None:
    x1, y1, x2, y2 = map(int, detection.bbox.to_list())
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    crop = frame[
        y1 + int((y2 - y1) * 0.18):y1 + int((y2 - y1) * 0.70),
        x1 + int((x2 - x1) * 0.18):x1 + int((x2 - x1) * 0.82),
    ]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 35, 20), (100, 255, 255))
    mask = cv2.bitwise_not(mask)
    if cv2.countNonZero(mask) < 20:
        mask = None
    histogram = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    flattened = histogram.flatten().astype("float32")
    norm = float((flattened @ flattened) ** 0.5)
    if norm <= 1e-8:
        return None
    return tuple(float(value) for value in flattened / norm)


def _canonicalize(detections: list[Detection]) -> list[Detection]:
    output: list[Detection] = []
    for detection in detections:
        detection.class_name = canonical_class_name(detection.class_name)
        output.append(detection)
    return output


def _overlap_count(
    left: list[Detection], right: list[Detection], class_name: str, threshold: float = 0.5
) -> int:
    candidates = sorted(
        (
            a.bbox.iou(b.bbox),
            left_index,
            right_index,
        )
        for left_index, a in enumerate(left)
        for right_index, b in enumerate(right)
        if a.class_name == class_name and b.class_name == class_name
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches = 0
    for overlap, left_index, right_index in reversed(candidates):
        if overlap < threshold:
            break
        if left_index not in used_left and right_index not in used_right:
            used_left.add(left_index)
            used_right.add(right_index)
            matches += 1
    return matches


def _draw(
    frame: Any,
    tracks: list[Track],
    detections: list[Detection],
    ball_assignment: dict[str, Any] | None,
    camera_motion: dict[str, Any] | None,
    team_model_ready: bool,
) -> Any:
    annotated = frame.copy()
    for detection in detections:
        if detection.class_name == "ball":
            x1, y1, x2, y2 = map(int, detection.bbox.to_list())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["ball"], 2)
            cv2.putText(
                annotated,
                f"ball {detection.confidence:.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS["ball"],
                2,
            )

    active = [track for track in tracks if track.missed == 0]
    nearest_id = (
        ball_assignment.get("ball_nearest_player")
        if ball_assignment is not None
        else None
    )
    for track in active:
        color = COLORS.get(track.class_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, track.bbox.to_list())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        team = track.metadata.get("team")
        label = f"ID {track.track_id}"
        if team:
            label += f" {team}"
        label += f" {track.class_name} {track.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            2,
        )
    if nearest_id is not None:
        nearest = next((track for track in active if track.track_id == nearest_id), None)
        ball = next((detection for detection in detections if detection.class_name == "ball"), None)
        if nearest is not None and ball is not None:
            cv2.line(
                annotated,
                tuple(map(int, ball.bbox.center)),
                tuple(map(int, nearest.bbox.footpoint)),
                (255, 255, 0),
                2,
            )
    cv2.putText(
        annotated,
        "PITCH: unavailable (no local keypoint model)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2,
    )
    motion_status = (
        f"Camera motion: {camera_motion.get('method')}"
        if camera_motion and camera_motion.get("valid")
        else "Camera motion: unavailable"
    )
    cv2.putText(
        annotated,
        motion_status,
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    if not team_model_ready:
        cv2.putText(
            annotated,
            "Teams: warming up",
            (20, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return annotated


class TrackingSummary:
    def __init__(self, frame_stride: int = 1) -> None:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        self.frame_stride = frame_stride
        self.tracks: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.class_track_counts: Counter[str] = Counter()

    def update(self, frame_index: int, tracks: list[Track]) -> None:
        active = [track for track in tracks if track.missed == 0]
        for track in tracks:
            state = self.tracks.setdefault(
                track.track_id,
                {
                    "class_name": track.class_name,
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "visible_frames": [],
                    "segments": 0,
                    "was_visible": False,
                    "last_visible_frame": None,
                    "reappearances": 0,
                    "team_changes": 0,
                    "teams_seen": [],
                    "team_confidences": [],
                    "team_uncertain_observations": 0,
                    "last_team_uncertain": True,
                    "last_center": None,
                    "last_appearance": None,
                    "max_jump_px_per_frame": 0.0,
                },
            )
            state["last_frame"] = frame_index
            is_visible = track.missed == 0
            if is_visible:
                if not state["was_visible"]:
                    state["segments"] += 1
                    if state["last_visible_frame"] is not None:
                        gap = frame_index - state["last_visible_frame"] - 1
                        if gap > 0:
                            state["reappearances"] += 1
                            self.events.append(
                                {
                                    "type": "track_reappearance",
                                    "frame_index": frame_index,
                                    "track_id": track.track_id,
                                    "missed_sample_intervals": max(
                                        0, gap // self.frame_stride - 1
                                    ),
                                    "source_frame_gap": gap,
                                    "ground_truth_switch": False,
                                }
                            )
                center = track.bbox.center
                previous_center = state["last_center"]
                previous_frame = state["last_visible_frame"]
                if previous_center is not None and previous_frame is not None:
                    frame_gap = max(1, frame_index - previous_frame)
                    dx = center[0] - previous_center[0]
                    dy = center[1] - previous_center[1]
                    speed = (dx * dx + dy * dy) ** 0.5 / frame_gap
                    state["max_jump_px_per_frame"] = max(
                        state["max_jump_px_per_frame"], speed
                    )
                    if speed > 180:
                        self.events.append(
                            {
                                "type": "large_position_jump",
                                "frame_index": frame_index,
                                "track_id": track.track_id,
                                "pixels_per_frame": round(speed, 2),
                                "ground_truth_switch": False,
                            }
                        )
                team = track.metadata.get("team_id")
                if team:
                    if state["teams_seen"] and state["teams_seen"][-1] != team:
                        state["team_changes"] += 1
                        self.events.append(
                            {
                                "type": "team_reclassification",
                                "frame_index": frame_index,
                                "track_id": track.track_id,
                                "from": state["teams_seen"][-1],
                                "to": team,
                                "ground_truth_switch": False,
                            }
                        )
                    if not state["teams_seen"] or state["teams_seen"][-1] != team:
                        state["teams_seen"].append(team)
                    confidence = track.metadata.get("team_confidence")
                    if confidence is not None:
                        state["team_confidences"].append(float(confidence))
                    if track.metadata.get("team_uncertain", False):
                        state["team_uncertain_observations"] += 1
                    state["last_team_uncertain"] = track.metadata.get(
                        "team_uncertain", False
                    )
                state["visible_frames"].append(frame_index)
                state["last_visible_frame"] = frame_index
                state["last_center"] = center
                state["last_appearance"] = track.appearance
                state["was_visible"] = True
            else:
                state["was_visible"] = False
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                if left.class_name == right.class_name and left.bbox.iou(right.bbox) > 0.6:
                    self.events.append(
                        {
                            "type": "overlapping_tracks_possible_merge",
                            "frame_index": frame_index,
                            "track_ids": [left.track_id, right.track_id],
                            "ground_truth_switch": False,
                        }
                    )

    def report(self) -> dict[str, Any]:
        durations = [
            len(state["visible_frames"])
            for state in self.tracks.values()
            if state["visible_frames"]
        ]
        return {
            "number_of_track_ids": len(self.tracks),
            "visible_observation_count_per_track": {
                "min": min(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0,
                "mean": round(statistics.mean(durations), 3) if durations else 0,
                "max": max(durations) if durations else 0,
            },
            "estimated_track_segments": sum(
                state["segments"] for state in self.tracks.values()
            ),
            "estimated_fragmentation_segments": sum(
                max(0, state["segments"] - 1) for state in self.tracks.values()
            ),
            "reappearance_events": sum(
                state["reappearances"] for state in self.tracks.values()
            ),
            "manual_verified_id_switches": None,
            "ground_truth_id_switches": None,
            "suspicious_event_count": len(self.events),
            "suspicious_events_are_not_ground_truth": True,
            "team_reclassification_events": sum(
                state["team_changes"] for state in self.tracks.values()
            ),
            "team_classification": {
                "player_track_count": sum(
                    state["class_name"] == "player" for state in self.tracks.values()
                ),
                "team_a_track_count": sum(
                    bool(
                        state["class_name"] == "player"
                        and state["teams_seen"]
                        and state["teams_seen"][-1] == TEAM_A
                    )
                    for state in self.tracks.values()
                ),
                "team_b_track_count": sum(
                    bool(
                        state["class_name"] == "player"
                        and state["teams_seen"]
                        and state["teams_seen"][-1] == TEAM_B
                    )
                    for state in self.tracks.values()
                ),
                "unclassified_track_count": sum(
                    state["class_name"] == "player" and not state["teams_seen"]
                    for state in self.tracks.values()
                ),
                "team_uncertain_track_count": sum(
                    bool(
                        state["class_name"] == "player"
                        and state["teams_seen"]
                        and state["last_team_uncertain"]
                    )
                    for state in self.tracks.values()
                ),
            },
            "tracks": {
                str(track_id): {
                    "class_name": state["class_name"],
                    "first_frame": state["first_frame"],
                    "last_frame": state["last_frame"],
                    "visible_sample_observations": len(state["visible_frames"]),
                    "duration_source_frames": (
                        state["last_visible_frame"] - state["first_frame"] + 1
                        if state["last_visible_frame"] is not None
                        else 0
                    ),
                    "segments": state["segments"],
                    "reappearances": state["reappearances"],
                    "teams_seen": state["teams_seen"],
                    "team_changes": state["team_changes"],
                    "final_team_id": (
                        state["teams_seen"][-1] if state["teams_seen"] else None
                    ),
                    "team_switch_count": state["team_changes"],
                    "mean_team_confidence": (
                        round(statistics.mean(state["team_confidences"]), 4)
                        if state["team_confidences"]
                        else None
                    ),
                    "team_confidence_std": (
                        round(statistics.pstdev(state["team_confidences"]), 4)
                        if len(state["team_confidences"]) > 1
                        else 0.0
                        if state["team_confidences"]
                        else None
                    ),
                    "team_uncertain_observations": state[
                        "team_uncertain_observations"
                    ],
                    "team_uncertain": state["last_team_uncertain"],
                    "max_jump_px_per_frame": round(
                        state["max_jump_px_per_frame"], 2
                    ),
                }
                for track_id, state in self.tracks.items()
            },
            "suspicious_events": self.events,
        }


def _make_pipeline(appearance_weight: float) -> Pipeline:
    tracker = TrackerConfig(
        max_age=30,
        association=AssociationConfig(
            iou_weight=0.35,
            center_weight=0.40,
            appearance_weight=appearance_weight,
            max_center_distance=140.0,
            max_cost=0.95,
        ),
    )
    return Pipeline(
        config=PipelineConfig(
            tracker=tracker,
            min_confidence=0.15,
            nms_iou_threshold=0.5,
        )
    )


def evaluate_video(
    video_path: Path,
    primary_model_path: Path,
    ball_model_path: Path,
    output_directory: Path,
    confidence: float = 0.15,
    team_warmup_frames: int = 20,
    frame_stride: int = 10,
    cpu_threads: int = 2,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not primary_model_path.is_file() or not ball_model_path.is_file():
        raise FileNotFoundError("Both configured detector weight files must exist")
    if frame_stride < 1 or cpu_threads < 1 or team_warmup_frames < 1:
        raise ValueError("frame_stride, cpu_threads, and team_warmup_frames must be positive")

    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    output_directory.mkdir(parents=True, exist_ok=True)
    debug_directory = output_directory / "debug"
    debug_directory.mkdir(parents=True, exist_ok=True)
    annotated_path = output_directory / "annotated_video.mp4"
    debug_video_path = debug_directory / "annotated_video.mp4"
    tracking_path = output_directory / "tracking_data.jsonl"
    possession_path = output_directory / "possession_data.jsonl"
    events_path = output_directory / "events.jsonl"
    summary_path = output_directory / "match_summary.json"

    primary = UltralyticsBackend(
        str(primary_model_path), conf=confidence, verbose=False
    )
    ball_backend = UltralyticsBackend(
        str(ball_model_path), conf=confidence, verbose=False
    )
    fusion = ModelAwareFusion()
    primary_names = getattr(primary.model, "names", {})
    ball_names = getattr(ball_backend.model, "names", {})
    role_decisions = {
        "player": "primary only: best.pt",
        "goalkeeper": "primary only: best.pt",
        "referee": "primary only: best.pt",
        "ball": "ball_referee.pt preferred; best.pt fallback only if absent",
        "secondary_people": "validation-only; not merged without labeled evidence",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_stride,
        (width, height),
    )
    debug_writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_stride,
        (width, height),
    )
    if not writer.isOpened() or not debug_writer.isOpened():
        cap.release()
        writer.release()
        debug_writer.release()
        raise RuntimeError("Could not create clean/debug annotated videos")

    enhanced_pipeline = _make_pipeline(appearance_weight=0.30)
    baseline_pipeline = _make_pipeline(appearance_weight=0.0)
    enhanced_summary = TrackingSummary(frame_stride)
    baseline_summary = TrackingSummary(frame_stride)
    team_classifier: TeamClassifier | None = None
    warmup_features: list[tuple[float, ...]] = []
    team_feature_history = TeamFeatureHistory(window_size=10)
    team_stabilizer = TeamStabilizer()
    team_classifier_failure: str | None = None
    team_classifier_attempted = False
    team_possession_samples: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    raw_counts: dict[str, Counter[str]] = {
        "primary": Counter(),
        "ball_referee": Counter(),
        "selected": Counter(),
    }
    confidence_values: dict[str, list[float]] = {}
    primary_secondary_matches: Counter[str] = Counter()
    primary_secondary_candidates: Counter[str] = Counter()
    duplicate_suppressions: Counter[str] = Counter()
    selected_confidences: dict[str, list[float]] = {}
    ball_indices: list[int] = []
    ball_gaps: list[int] = []
    last_ball_frame: int | None = None
    ball_jump_count = 0
    ball_distances: list[float] = []
    camera_valid = 0
    camera_association_frames = 0
    camera_compensated_track_observations = 0
    camera_magnitudes: list[float] = []
    frame_index = 0
    sampled_frames = 0
    source_frame_index = 0
    started = time.perf_counter()
    processing_seconds = 0.0
    nms_filter = DetectionFilter(
        min_confidence=confidence,
        nms_iou_threshold=0.5,
        person_duplicate_containment_threshold=0.75,
        person_duplicate_footpoint_distance_ratio=0.20,
        person_duplicate_center_distance_ratio=0.50,
        person_duplicate_min_height_ratio=0.55,
        person_duplicate_max_appearance_distance=0.15,
    )

    try:
        with (
            tracking_path.open("w", encoding="utf-8") as tracking_file,
            possession_path.open("w", encoding="utf-8") as possession_file,
            events_path.open("w", encoding="utf-8") as events_file,
        ):
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = source_frame_index
                source_frame_index += 1
                if frame_index % frame_stride != 0:
                    continue
                frame_started = time.perf_counter()
                primary_detections = _canonicalize(primary(frame))
                ball_model_detections = _canonicalize(ball_backend(frame))
                for detection in primary_detections:
                    raw_counts["primary"][detection.class_name] += 1
                    confidence_values.setdefault(
                        f"primary:{detection.class_name}", []
                    ).append(detection.confidence)
                for detection in ball_model_detections:
                    raw_counts["ball_referee"][detection.class_name] += 1
                    confidence_values.setdefault(
                        f"ball_referee:{detection.class_name}", []
                    ).append(detection.confidence)

                for class_name in ("player", "referee"):
                    primary_secondary_candidates[class_name] += sum(
                        detection.class_name == class_name
                        for detection in primary_detections
                    )
                    primary_secondary_matches[class_name] += _overlap_count(
                        primary_detections,
                        ball_model_detections,
                        class_name,
                    )
                routed = fusion(
                    {
                        "primary": primary_detections,
                        "ball_referee": ball_model_detections,
                    }
                )
                for detection in routed:
                    if detection.class_name in {"player", "goalkeeper"}:
                        feature = _feature(frame, detection)
                        detection.appearance = feature

                visible_detections = nms_filter(routed)
                input_counts = Counter(
                    detection.class_name for detection in routed
                )
                kept_counts = Counter(
                    detection.class_name for detection in visible_detections
                )
                for class_name, count in input_counts.items():
                    duplicate_suppressions[class_name] += max(
                        0, count - kept_counts[class_name]
                    )
                if team_classifier is None and (
                    frame_index < team_warmup_frames
                    or (
                        not team_classifier_attempted
                        and len(warmup_features) < 20
                    )
                ):
                    for detection in visible_detections:
                        if (
                            detection.class_name != "player"
                            or detection.confidence < 0.45
                            or detection.bbox.x1 <= 2
                            or detection.bbox.x2 >= width - 2
                            or detection.bbox.y1 <= 2
                        ):
                            continue
                        feature = extract_team_color_feature(frame, detection.bbox)
                        if feature is not None:
                            warmup_features.append(feature)
                if (
                    team_classifier is None
                    and not team_classifier_attempted
                    and frame_index + 1 >= team_warmup_frames
                    and len(warmup_features) >= 20
                ):
                    team_classifier_attempted = True
                    try:
                        team_classifier = TeamClassifier().fit(warmup_features)
                    except ValueError as error:
                        team_classifier_failure = str(error)
                raw_counts["selected"].update(
                    detection.class_name for detection in visible_detections
                )
                for detection in visible_detections:
                    selected_confidences.setdefault(detection.class_name, []).append(
                        detection.confidence
                    )

                baseline_detections = [
                    Detection(
                        detection.bbox,
                        detection.confidence,
                        detection.class_name,
                        metadata=dict(detection.metadata),
                    )
                    for detection in visible_detections
                ]
                baseline_record = baseline_pipeline.process_frame(
                    {"detections": baseline_detections}, frame_index
                )
                enhanced_record = enhanced_pipeline.process_frame(
                    {"image": frame, "detections": visible_detections},
                    frame_index,
                )
                for track in enhanced_record.tracks:
                    if track.class_name != "player":
                        continue
                    if track.missed != 0 or team_classifier is None:
                        track.metadata.setdefault("team_id", None)
                        track.metadata.setdefault("team_confidence", 0.0)
                        track.metadata.setdefault("team_uncertain", True)
                        track.metadata.setdefault("team_history", [])
                        track.metadata.setdefault("team_observations", 0)
                        continue
                    team_feature = extract_team_color_feature(frame, track.bbox)
                    if team_feature is None:
                        continue
                    try:
                        raw_cluster, raw_confidence = (
                            team_classifier.predict_with_confidence(team_feature)
                        )
                        smoothed_feature = team_feature_history.update(
                            track.track_id,
                            team_feature,
                        )
                        predicted_cluster, prediction_confidence = (
                            team_classifier.predict_with_confidence(
                                smoothed_feature
                            )
                        )
                        if (
                            raw_confidence >= TEAM_FEATURE_RESET_CONFIDENCE
                            and raw_cluster != predicted_cluster
                        ):
                            smoothed_feature = team_feature_history.reset(
                                track.track_id,
                                team_feature,
                            )
                            predicted_cluster, prediction_confidence = (
                                team_classifier.predict_with_confidence(
                                    smoothed_feature
                                )
                            )
                    except ValueError as error:
                        raise RuntimeError(
                            f"Could not classify team for track {track.track_id} "
                            f"at frame {frame_index}"
                        ) from error
                    semantic_team = TEAM_A if predicted_cluster == 0 else TEAM_B
                    track.metadata.update(
                        team_stabilizer.update(
                            track.track_id,
                            semantic_team,
                            prediction_confidence,
                        )
                    )
                    track.metadata["team"] = track.metadata["team_id"]
                    track.metadata["team_name"] = None
                enhanced_summary.update(frame_index, enhanced_record.tracks)
                baseline_summary.update(frame_index, baseline_record.tracks)

                assignment = enhanced_record.ball_assignment or {}
                if assignment.get("ball_nearest_player") is not None:
                    nearest_id = int(assignment["ball_nearest_player"])
                    nearest_track = next(
                        (
                            track for track in enhanced_record.tracks
                            if track.track_id == nearest_id and track.missed == 0
                        ),
                        None,
                    )
                    ball_detection = next(
                        (
                            detection for detection in visible_detections
                            if detection.class_name == "ball"
                        ),
                        None,
                    )
                    if nearest_track is not None and ball_detection is not None:
                        ball_distances.append(
                            (
                                (ball_detection.bbox.center[0] - nearest_track.bbox.footpoint[0]) ** 2
                                + (ball_detection.bbox.center[1] - nearest_track.bbox.footpoint[1]) ** 2
                            ) ** 0.5
                        )

                balls = [
                    detection for detection in visible_detections
                    if detection.class_name == "ball"
                ]
                if balls:
                    ball_indices.append(frame_index)
                    if last_ball_frame is not None:
                        missed_sample_intervals = (
                            frame_index - last_ball_frame
                        ) // frame_stride - 1
                        if missed_sample_intervals > 0:
                            ball_gaps.append(missed_sample_intervals)
                    ball = max(balls, key=lambda item: item.confidence)
                    prior_ball = getattr(evaluate_video, "_prior_ball", None)
                    if prior_ball is not None:
                        prior_index, prior_position = prior_ball
                        delta = frame_index - prior_index
                        distance = (
                            (ball.bbox.center[0] - prior_position[0]) ** 2
                            + (ball.bbox.center[1] - prior_position[1]) ** 2
                        ) ** 0.5
                        if delta <= 5 * frame_stride and distance / delta > 250:
                            ball_jump_count += 1
                    evaluate_video._prior_ball = (frame_index, ball.bbox.center)
                    last_ball_frame = frame_index

                motion = enhanced_record.camera_motion or {}
                if motion.get("association_used"):
                    camera_association_frames += 1
                camera_compensated_track_observations += int(
                    motion.get("association_compensated_track_count", 0)
                )
                if motion.get("valid"):
                    camera_valid += 1
                    matrix = motion.get("matrix")
                    if matrix:
                        camera_magnitudes.append(
                            (float(matrix[0][2]) ** 2 + float(matrix[1][2]) ** 2) ** 0.5
                        )
                record = enhanced_record.to_dict()
                # The visual descriptor is used internally for association but is
                # not part of the public tracking record.
                for detection_data in record["detections"]:
                    detection_data["appearance"] = None
                for track_data in record["tracks"]:
                    track_data["appearance"] = None
                    track_data["metadata"]["appearance_feature_present"] = (
                        next(
                            (
                                track.appearance is not None
                                for track in enhanced_record.tracks
                                if track.track_id == track_data["track_id"]
                            ),
                            False,
                        )
                    )
                tracking_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                possession_state = record.get("possession") or {}
                if possession_state.get("possessor_team"):
                    team_possession_samples[
                        str(possession_state["possessor_team"])
                    ] += 1
                possession_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "possession": possession_state,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                for event in record.get("events", []):
                    event_counts[str(event["type"])] += 1
                    events_file.write(
                        json.dumps(event, separators=(",", ":")) + "\n"
                    )

                writer.write(draw_tracking_frame(frame, record))
                debug_writer.write(draw_debug_frame(frame, record))
                processing_seconds += time.perf_counter() - frame_started
                sampled_frames += 1
                if sampled_frames % 25 == 0:
                    print(
                        f"processed {sampled_frames} samples "
                        f"(source frame {frame_index}/{total_frames})",
                        flush=True,
                    )
    finally:
        cap.release()
        writer.release()
        debug_writer.release()
        if hasattr(evaluate_video, "_prior_ball"):
            delattr(evaluate_video, "_prior_ball")

    elapsed = time.perf_counter() - started
    enhanced_report = enhanced_summary.report()
    team_report = enhanced_report["team_classification"]
    team_report.update(
        {
            "team_ids": [TEAM_A, TEAM_B],
            "team_labeling_basis": (
                "Teams are ordered by dominant torso BGR centroid; red is TEAM_A "
                "and white is TEAM_B in vid002. Labels are generic within-match "
                "identities, not real-world team names."
            ),
            "warmup_player_features": len(warmup_features),
            "classifier_status": (
                "fit" if team_classifier is not None else "unavailable"
            ),
            "classifier_failure_reason": team_classifier_failure,
            "track_assignments": team_stabilizer.report(),
        }
    )
    possession_report = enhanced_pipeline.possession.report()
    possession_observations = sum(team_possession_samples.values())
    match_summary = {
        "video": video_path.name,
        "frames_decoded": source_frame_index,
        "sampled_frames": sampled_frames,
        "sampling_stride": frame_stride,
        "fps": fps,
        "decoded_duration_seconds": (
            source_frame_index / fps if fps > 0 else None
        ),
        "processing_seconds": round(elapsed, 3),
        "tracking": {
            "track_count": enhanced_report["number_of_track_ids"],
            "estimated_fragmentation_segments": enhanced_report[
                "estimated_fragmentation_segments"
            ],
            "ground_truth_id_switches": None,
        },
        "team_classification": team_report,
        "possession": {
            **possession_report,
            "confirmed_samples_by_team": dict(team_possession_samples),
            "confirmed_sample_share_by_team": {
                team: round(count / possession_observations, 4)
                for team, count in team_possession_samples.items()
            }
            if possession_observations
            else {},
            "unclassified_or_unobserved_samples": (
                sampled_frames - possession_observations
            ),
        },
        "events": dict(event_counts),
        "limitations": [
            "No labeled ground truth is available for detector or identity accuracy.",
            "Passes, carries, shots, and event subtypes are not inferred without sufficient ball/pitch evidence.",
            "No local field-keypoint model is configured; pitch-space possession is unavailable.",
        ],
    }
    summary_path.write_text(
        json.dumps(match_summary, indent=2),
        encoding="utf-8",
    )
    detection_report = {
        "frames_processed": sampled_frames,
        "container_reported_frame_count": total_frames,
        "source_frames_decoded": source_frame_index,
        "decoder_stopped_after_read_failure": True,
        "metadata_frame_count_matches_decoded_frames": (
            source_frame_index == total_frames
        ),
        "sampling": {
            "frame_stride": frame_stride,
            "sampled_frames": sampled_frames,
            "source_frames_covered": source_frame_index,
            "coverage_fraction_of_decodable_video": 1.0,
            "inference_on_every_source_frame": frame_stride == 1,
            "annotated_video_fps": round(fps / frame_stride, 4),
        },
        "class_counts_by_model": {
            key: dict(value) for key, value in raw_counts.items()
        },
        "confidence_statistics_by_model_and_class": {
            key: {
                "count": len(values),
                "mean": round(statistics.mean(values), 4) if values else None,
                "median": round(statistics.median(values), 4) if values else None,
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
            }
            for key, values in confidence_values.items()
        },
        "selected_confidence_statistics": {
            key: {
                "count": len(values),
                "mean": round(statistics.mean(values), 4) if values else None,
                "median": round(statistics.median(values), 4) if values else None,
            }
            for key, values in selected_confidences.items()
        },
        "cross_model_iou_matches_at_0_5": dict(primary_secondary_matches),
        "primary_validation_denominators": dict(primary_secondary_candidates),
        "duplicate_filter": {
            "class_aware_iou_threshold": 0.5,
            "contained_box_threshold": 0.90,
            "person_duplicate_containment_threshold": 0.75,
            "person_duplicate_footpoint_distance_ratio": 0.20,
            "person_duplicate_center_distance_ratio": 0.50,
            "person_duplicate_min_height_ratio": 0.55,
            "person_duplicate_max_appearance_distance": 0.15,
            "removed_detections_by_class": dict(duplicate_suppressions),
            "note": (
                "Same-class boxes are suppressed by IoU or containment; person "
                "boxes also require nearby footpoints for the 75% containment rule."
            ),
        },
        "outside_pitch_detections": None,
        "missing_detections": None,
        "ground_truth_available": False,
        "model_roles": role_decisions,
        "model_class_names": {
            "best.pt": {str(key): value for key, value in primary_names.items()},
            "ball_referee.pt": {
                str(key): value for key, value in ball_names.items()
            },
        },
        "confidence_threshold": confidence,
    }
    ball_detection_rate = (
        len(ball_indices) / sampled_frames if sampled_frames else 0.0
    )
    ball_report = {
        "detected_frames": len(ball_indices),
        "detection_rate": round(ball_detection_rate, 5),
        "sampled_frames_without_ball": sampled_frames - len(ball_indices),
        "sampling_stride": frame_stride,
        "gaps_between_detections_in_sample_intervals": len(ball_gaps),
        "longest_gap_sample_intervals": max(ball_gaps) if ball_gaps else 0,
        "mean_gap_sample_intervals": (
            round(statistics.mean(ball_gaps), 3) if ball_gaps else 0
        ),
        "source_frames_between_sampled_detections": (
            frame_stride * (max(ball_gaps) + 1) if ball_gaps else 0
        ),
        "candidate_large_jumps": ball_jump_count,
        "jump_rule": "pixel displacement >250 px/frame for gaps of at most 5 frames; suspicious only, not ground truth",
        "mean_ball_to_nearest_player_distance_px": (
            round(statistics.mean(ball_distances), 2) if ball_distances else None
        ),
        "ball_assignment_is_nearest_player_not_possession": True,
    }
    tracking_report = {
        "enhanced_appearance_association": enhanced_report,
        "baseline_without_appearance": baseline_summary.report(),
        "ablation": {
            "same_detections_and_geometry": True,
            "appearance_enabled_weight": 0.30,
            "baseline_appearance_weight": 0.0,
            "comparison_basis": "automatically detected track reappearance, fragmentation, overlap, and jump anomalies only; no ground-truth identity labels",
        },
        "team_classification": team_report,
        "pitch_geometry": {
            "status": "not_configured",
            "reason": "No local field-keypoint weights are available; external inference was not called.",
            "keypoint_detection_rate": None,
            "homography_valid_frames": 0,
            "reprojection_error": None,
        },
        "camera_motion": {
            "valid_frames": camera_valid,
            "valid_sample_rate": (
                round(camera_valid / sampled_frames, 5) if sampled_frames else 0.0
            ),
            "mean_translation_px": (
                round(statistics.mean(camera_magnitudes), 3)
                if camera_magnitudes else None
            ),
            "association_frames": camera_association_frames,
            "compensated_track_observations": camera_compensated_track_observations,
            "used_for_tracking_association": camera_association_frames > 0,
        },
        "performance": {
            "processing_seconds_including_inference_and_render": round(elapsed, 3),
            "processing_seconds_per_frame": (
                round(processing_seconds / sampled_frames, 5)
                if sampled_frames else None
            ),
            "end_to_end_sampled_fps": (
                round(sampled_frames / elapsed, 3) if elapsed else None
            ),
            "sampled_frames_per_second": (
                round(sampled_frames / elapsed, 3) if elapsed else None
            ),
            "source_frames_per_second_equivalent": (
                round(source_frame_index / elapsed, 3) if elapsed else None
            ),
            "video_fps": fps,
            "cpu_threads_limited_to": cpu_threads,
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
    (output_directory / "detection_metrics.json").write_text(
        json.dumps(detection_report, indent=2), encoding="utf-8"
    )
    (output_directory / "tracking_metrics.json").write_text(
        json.dumps(tracking_report, indent=2), encoding="utf-8"
    )
    (debug_directory / "suspicious_events.json").write_text(
        json.dumps(
            {
                "enhanced": enhanced_summary.events,
                "baseline": baseline_summary.events,
                "id_switches_are_not_ground_truth": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (debug_directory / "ball_metrics.json").write_text(
        json.dumps(ball_report, indent=2), encoding="utf-8"
    )
    (debug_directory / "model_fusion.json").write_text(
        json.dumps(
            {
                "roles": role_decisions,
                "class_names": detection_report["model_class_names"],
                "cross_model_iou_matches_at_0_5": dict(primary_secondary_matches),
                "primary_validation_denominators": dict(
                    primary_secondary_candidates
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "frames_processed": sampled_frames,
        "source_frames_decoded": source_frame_index,
        "container_reported_frames": total_frames,
        "sampling_stride": frame_stride,
        "annotated_video": str(annotated_path),
        "debug_video": str(debug_video_path),
        "tracking_data": str(tracking_path),
        "detection_metrics": str(output_directory / "detection_metrics.json"),
        "tracking_metrics": str(output_directory / "tracking_metrics.json"),
        "elapsed_seconds": round(elapsed, 2),
        "sampled_fps": (
            round(sampled_frames / elapsed, 2) if elapsed else None
        ),
        "selected_detection_counts": dict(raw_counts["selected"]),
        "ball_detection_rate": round(ball_detection_rate, 4),
        "possession_data": str(possession_path),
        "events": str(events_path),
        "match_summary": str(summary_path),
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("..") / "football-yolo_1" / "data" / "videos" / "vid002.mp4",
    )
    parser.add_argument(
        "--primary-model",
        type=Path,
        default=Path("..") / "football-yolo_1" / "models" / "best.pt",
    )
    parser.add_argument(
        "--ball-model",
        type=Path,
        default=Path("..") / "football-yolo_1" / "models" / "ball_referee.pt",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs") / "vid002")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--team-warmup-frames", type=int, default=20)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="Infer every Nth source frame; 10 limits CPU load while spanning the full clip.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="Limit PyTorch CPU intra-op threads for a lower-impact run.",
    )
    args = parser.parse_args(argv)
    try:
        evaluate_video(
            args.video,
            args.primary_model,
            args.ball_model,
            args.output,
            confidence=args.confidence,
            team_warmup_frames=args.team_warmup_frames,
            frame_stride=args.frame_stride,
            cpu_threads=args.cpu_threads,
        )
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"video evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
