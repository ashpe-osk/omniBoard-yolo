"""Render clean and debug videos from an existing tracking JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from footy_cvision.visualization.annotations import (
    draw_debug_frame,
    draw_tracking_frame,
)


def _read_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame_index = int(record["frame_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid tracking record at line {line_number}"
                ) from error
            if frame_index in records:
                raise ValueError(f"Duplicate frame_index {frame_index} in {path}")
            records[frame_index] = record
    return records


def render_tracking_video(
    video_path: Path,
    records_path: Path,
    output_directory: Path,
    frame_stride: int = 10,
) -> tuple[Path, Path]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    records = _read_records(records_path)
    if not records:
        raise ValueError("Tracking JSONL contains no records")
    output_directory.mkdir(parents=True, exist_ok=True)
    debug_directory = output_directory / "debug"
    debug_directory.mkdir(parents=True, exist_ok=True)
    clean_path = output_directory / "annotated_video.mp4"
    debug_path = debug_directory / "annotated_video.mp4"

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = source_fps / frame_stride
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    clean_writer = cv2.VideoWriter(
        str(clean_path), codec, output_fps, (width, height)
    )
    debug_writer = cv2.VideoWriter(
        str(debug_path), codec, output_fps, (width, height)
    )
    if not clean_writer.isOpened() or not debug_writer.isOpened():
        capture.release()
        clean_writer.release()
        debug_writer.release()
        raise RuntimeError("Could not open both clean and debug video writers")

    decoded_frames = 0
    rendered_frames = 0
    missing_record_frames: list[int] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_frames % frame_stride == 0:
                record = records.get(decoded_frames)
                if record is None:
                    missing_record_frames.append(decoded_frames)
                else:
                    clean_writer.write(draw_tracking_frame(frame, record))
                    debug_writer.write(draw_debug_frame(frame, record))
                    rendered_frames += 1
            decoded_frames += 1
    finally:
        capture.release()
        clean_writer.release()
        debug_writer.release()

    unexpected_records = sorted(set(records) - set(range(0, decoded_frames, frame_stride)))
    if unexpected_records:
        raise ValueError(
            f"Tracking records do not align to source sampling: {unexpected_records[:5]}"
        )
    if missing_record_frames:
        raise ValueError(
            f"Missing tracking records for sampled frames: {missing_record_frames[:5]}"
        )
    print(
        json.dumps(
            {
                "decoded_source_frames": decoded_frames,
                "tracking_records": len(records),
                "rendered_frames": rendered_frames,
                "frame_stride": frame_stride,
                "video_fps": source_fps,
                "output_fps": output_fps,
                "clean_video": str(clean_path),
                "debug_video": str(debug_path),
            },
            indent=2,
        )
    )
    return clean_path, debug_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracking-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        render_tracking_video(
            args.video,
            args.tracking_data,
            args.output,
            args.frame_stride,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"video rendering failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
