"""Small, dependency-free evaluation metrics for frame records."""

from __future__ import annotations

from math import hypot
from typing import Any, Iterable, Sequence

from footy_cvision.core import Detection
from footy_cvision.detection.filter import bbox_from


def iou_match(
    predictions: Sequence[Detection],
    ground_truths: Sequence[Detection],
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    candidates = sorted(
        (
            predictions[prediction_index].bbox.iou(ground_truth.bbox),
            prediction_index,
            truth_index,
        )
        for prediction_index, prediction in enumerate(predictions)
        for truth_index, ground_truth in enumerate(ground_truths)
        if prediction.class_name == ground_truth.class_name
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    for overlap, prediction_index, truth_index in reversed(candidates):
        if overlap < threshold:
            break
        if prediction_index not in used_predictions and truth_index not in used_truths:
            used_predictions.add(prediction_index)
            used_truths.add(truth_index)
    true_positive = len(used_predictions)
    return true_positive, len(predictions) - true_positive, len(ground_truths) - true_positive


def detection_metrics(
    predictions: Sequence[Sequence[Detection]],
    ground_truths: Sequence[Sequence[Detection]],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    if len(predictions) != len(ground_truths):
        raise ValueError("Prediction and ground-truth sequences must align by frame")
    tp = fp = fn = 0
    for predicted, truth in zip(predictions, ground_truths):
        frame_tp, frame_fp, frame_fn = iou_match(predicted, truth, threshold)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall}


def evaluate(
    records: Sequence[dict[str, Any]],
    ground_truths: Sequence[Sequence[Detection]] | None = None,
    pitch_polygon: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compute available aggregate metrics from serialized pipeline records."""
    result: dict[str, Any] = {}
    if ground_truths is not None:
        predictions = [
            [
                Detection(
                    bbox=bbox_from(item["bbox"]),
                    confidence=float(item["confidence"]),
                    class_name=str(item["class_name"]),
                )
                for item in record.get("detections", [])
            ]
            for record in records
        ]
        result["detection"] = detection_metrics(predictions, ground_truths)

    gt_history: dict[int, list[tuple[int, int]]] = {}
    track_observations: dict[int, list[int]] = {}
    ball_positions: list[tuple[int, float, float]] = []
    speeds: list[float] = []
    outside = total_detections = 0
    for record_index, record in enumerate(records):
        frame_index = int(record.get("frame_index", record_index))
        for track in record.get("tracks", []):
            track_id = int(track["track_id"])
            if int(track.get("missed", 0)) == 0:
                track_observations.setdefault(track_id, []).append(frame_index)
                metadata = track.get("metadata", {})
                ground_truth_id = metadata.get("ground_truth_id")
                if ground_truth_id is not None:
                    gt_history.setdefault(int(ground_truth_id), []).append(
                        (frame_index, track_id)
                    )
        for detection in record.get("detections", []):
            total_detections += 1
            if pitch_polygon is not None:
                box = bbox_from(detection["bbox"])
                from footy_cvision.detection.filter import point_in_polygon

                if not point_in_polygon(box.footpoint, pitch_polygon):
                    outside += 1
            if detection.get("class_name") == "ball":
                box = bbox_from(detection["bbox"])
                ball_positions.append((frame_index, *box.center))
        for track in record.get("tracks", []):
            if int(track.get("missed", 0)) != 0:
                continue
            velocity = track.get("metadata", {}).get("speed")
            if velocity is not None:
                speeds.append(float(velocity))

    switches = fragmentation = 0
    for observations in gt_history.values():
        observations.sort()
        prior_id: int | None = None
        prior_frame: int | None = None
        was_visible = False
        for frame, track_id in observations:
            if prior_id is not None and track_id != prior_id:
                switches += 1
            if prior_frame is not None and frame > prior_frame + 1 and was_visible:
                fragmentation += 1
            prior_id, prior_frame, was_visible = track_id, frame, True
    persistence = [
        len(frames) / (max(frames) - min(frames) + 1)
        for frames in track_observations.values()
        if frames
    ]
    ball_steps = [
        hypot(current[1] - previous[1], current[2] - previous[2])
        for previous, current in zip(ball_positions, ball_positions[1:])
        if current[0] == previous[0] + 1
    ]
    result.update(
        {
            "id_switches": switches,
            "fragmentation": fragmentation,
            "mean_track_persistence": (
                sum(persistence) / len(persistence) if persistence else 0.0
            ),
            "ball_consistency_mean_displacement": (
                sum(ball_steps) / len(ball_steps) if ball_steps else None
            ),
            "outside_pitch_rate": outside / total_detections if total_detections else 0.0,
            "mean_speed": sum(speeds) / len(speeds) if speeds else None,
        }
    )
    return result
