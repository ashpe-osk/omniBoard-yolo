"""CLI for processing JSON/JSONL detections with an optional detector backend."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

from footy_cvision.pipeline.pipeline import (
    Pipeline,
    PipelineConfig,
    records_to_jsonable,
)
from footy_cvision.tracking.association import AssociationConfig
from footy_cvision.tracking.tracker import TrackerConfig


def load_config(path: Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required when using --config") from error
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a top-level mapping")
    pipeline = config.get("pipeline", {})
    tracker_data = config.get("tracker", {})
    association_data = config.get("association", {})
    pitch = pipeline.get("pitch_polygon")
    association = AssociationConfig(
        iou_weight=float(association_data.get("iou_weight", 0.45)),
        center_weight=float(association_data.get("center_weight", 0.35)),
        appearance_weight=float(association_data.get("appearance_weight", 0.20)),
        max_center_distance=float(association_data.get("max_center_distance", 100)),
        min_iou=float(association_data.get("min_iou", 0)),
        max_cost=float(association_data.get("max_cost", 0.95)),
    )
    return PipelineConfig(
        tracker=TrackerConfig(
            max_age=int(tracker_data.get("max_age", 10)),
            min_hits=int(tracker_data.get("min_hits", 1)),
            association=association,
        ),
        min_confidence=float(pipeline.get("min_confidence", 0.0)),
        ball_class=str(pipeline.get("ball_class", "ball")),
        player_classes=tuple(pipeline.get("player_classes", ["player"])),
        max_ball_distance=float(pipeline.get("max_ball_distance", 120.0)),
        ball_min_confidence=float(pipeline.get("ball_min_confidence", 0.0)),
        max_ball_distance_m=float(pipeline.get("max_ball_distance_m", 3.0)),
        possession_confirm_observations=int(
            pipeline.get("possession_confirm_observations", 2)
        ),
        possession_release_observations=int(
            pipeline.get("possession_release_observations", 2)
        ),
        pitch_polygon=(
            tuple(tuple(map(float, point)) for point in pitch) if pitch else None
        ),
        max_speed=float(pipeline.get("max_speed", 250.0)),
        nms_iou_threshold=(
            float(pipeline["nms_iou_threshold"])
            if pipeline.get("nms_iou_threshold") is not None
            else None
        ),
        containment_threshold=(
            float(pipeline["containment_threshold"])
            if pipeline.get("containment_threshold") is not None
            else None
        ),
        person_duplicate_containment_threshold=(
            float(pipeline["person_duplicate_containment_threshold"])
            if pipeline.get("person_duplicate_containment_threshold") is not None
            else None
        ),
        person_duplicate_footpoint_distance_ratio=(
            float(pipeline["person_duplicate_footpoint_distance_ratio"])
            if pipeline.get("person_duplicate_footpoint_distance_ratio") is not None
            else None
        ),
        person_duplicate_center_distance_ratio=(
            float(pipeline["person_duplicate_center_distance_ratio"])
            if pipeline.get("person_duplicate_center_distance_ratio") is not None
            else None
        ),
        person_duplicate_min_height_ratio=(
            float(pipeline["person_duplicate_min_height_ratio"])
            if pipeline.get("person_duplicate_min_height_ratio") is not None
            else None
        ),
        person_duplicate_max_appearance_distance=(
            float(pipeline["person_duplicate_max_appearance_distance"])
            if pipeline.get("person_duplicate_max_appearance_distance") is not None
            else None
        ),
    )


def load_detector(spec: str | None) -> Callable[[Any], Any] | None:
    if spec is None:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Detector must be specified as 'module:callable'")
    detector = getattr(importlib.import_module(module_name), attribute)
    if not callable(detector):
        raise TypeError(f"Detector backend {spec!r} is not callable")
    return detector


def read_frames(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("frames", [parsed] if "detections" in parsed else None)
    if not isinstance(parsed, list):
        raise ValueError("Input JSON must be a frame list or an object with 'frames'")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="JSON/JSONL frame detections"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    parser.add_argument("--config", type=Path, help="Optional YAML config file")
    parser.add_argument(
        "--detector",
        help="Optional detector callback as module:callable; called per frame",
    )
    args = parser.parse_args(argv)
    try:
        frames = read_frames(args.input)
        pipeline = Pipeline(load_detector(args.detector), load_config(args.config))
        results = pipeline.run(frames)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records_to_jsonable(results), indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError, TypeError, ImportError) as error:
        print(f"footy-cvision: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
