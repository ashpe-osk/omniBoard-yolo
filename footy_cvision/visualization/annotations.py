"""Clean and diagnostic annotations for football tracking records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from footy_cvision.core import BoundingBox

TEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "TEAM_A": (30, 40, 235),
    "TEAM_B": (240, 240, 240),
    "UNCLASSIFIED": (160, 160, 160),
}
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (40, 220, 40),
    "goalkeeper": (0, 165, 255),
    "referee": (220, 80, 220),
    "ball": (0, 220, 255),
}


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video annotation") from error
    return cv2


def _coordinates(
    bbox: BoundingBox | Sequence[float],
) -> tuple[float, float, float, float]:
    if isinstance(bbox, BoundingBox):
        return bbox.x1, bbox.y1, bbox.x2, bbox.y2
    if len(bbox) != 4:
        raise ValueError("A bounding box must have four coordinates")
    return tuple(map(float, bbox))  # type: ignore[return-value]


def draw_football_ellipse(
    frame: Any,
    bbox: BoundingBox | Sequence[float],
    color: tuple[int, int, int],
    track_id: int | str | None = None,
) -> Any:
    """Draw the reference football ellipse and centered ID badge.

    The arc center/radii and angle range mirror the reference tracker's
    ``draw_ellipse`` geometry while keeping this project self-contained.
    """
    cv2 = _cv2()
    x1, _, x2, y2 = _coordinates(bbox)
    x_center = int((x1 + x2) / 2)
    width = max(1, int(x2 - x1))
    y_bottom = int(y2)
    x_radius = width
    y_radius = max(1, int(0.35 * width))
    cv2.ellipse(
        frame,
        center=(x_center, y_bottom),
        axes=(x_radius, y_radius),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )
    if track_id is not None:
        text = str(track_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        padding_x, padding_y = 7, 4
        badge_width = text_size[0] + padding_x * 2
        badge_height = text_size[1] + baseline + padding_y * 2
        badge_top = y_bottom - badge_height // 2
        badge_left = x_center - badge_width // 2
        cv2.rectangle(
            frame,
            (badge_left, badge_top),
            (badge_left + badge_width, badge_top + badge_height),
            color,
            cv2.FILLED,
        )
        text_x = x_center - text_size[0] // 2
        text_y = y_bottom + (text_size[1] - baseline) // 2
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def _active_tracks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        track
        for track in record.get("tracks", [])
        if int(track.get("missed", 0)) == 0
    ]


def _draw_ball(
    frame: Any,
    detections: Sequence[Mapping[str, Any]],
    possession: Mapping[str, Any] | None,
    tracks: Sequence[Mapping[str, Any]],
) -> None:
    cv2 = _cv2()
    balls = [
        detection
        for detection in detections
        if detection.get("class_name") == "ball"
    ]
    if not balls:
        return
    ball = max(balls, key=lambda item: float(item.get("confidence", 0.0)))
    x1, y1, x2, y2 = _coordinates(ball["bbox"])
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    cv2.circle(frame, center, 8, CLASS_COLORS["ball"], 2, cv2.LINE_AA)
    cv2.drawMarker(
        frame,
        center,
        CLASS_COLORS["ball"],
        cv2.MARKER_DIAMOND,
        16,
        2,
        cv2.LINE_AA,
    )
    possessor_id = (possession or {}).get("possessor_id")
    if possessor_id is None:
        return
    player = next(
        (
            track
            for track in tracks
            if int(track["track_id"]) == int(possessor_id)
            and track.get("class_name") in {"player", "goalkeeper"}
        ),
        None,
    )
    if player is not None:
        foot_x = int((player["bbox"][0] + player["bbox"][2]) / 2)
        foot_y = int(player["bbox"][3])
        cv2.line(frame, center, (foot_x, foot_y), (255, 255, 0), 1, cv2.LINE_AA)


def _draw_status_panel(
    frame: Any,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
) -> None:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.55, max(0.32, width / 2000))
    thickness = 1
    padding = 10
    line_gap = 7
    text_sizes = [
        cv2.getTextSize(text, font, scale, thickness)
        for text, _ in lines
    ]
    line_heights = [size[0][1] + size[1] for size in text_sizes]
    panel_width = min(
        max(1, width - 20),
        max((size[0][0] for size in text_sizes), default=0) + padding * 2,
    )
    panel_height = sum(line_heights) + line_gap * max(0, len(lines) - 1) + padding * 2
    x1 = max(10, width - panel_width - 10)
    y1 = 12
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (min(width - 1, x1 + panel_width), min(height - 1, y1 + panel_height)),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    text_y = y1 + padding
    for (text, color), (text_size, baseline) in zip(lines, text_sizes):
        text_y += text_size[1]
        cv2.putText(
            frame,
            text,
            (x1 + padding, text_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_y += baseline + line_gap


def draw_tracking_frame(frame: Any, record: Mapping[str, Any]) -> Any:
    """Render semantic team colors, referee, ball, and confirmed possession."""
    cv2 = _cv2()
    output = frame.copy()
    active = _active_tracks(record)
    for track in active:
        class_name = str(track.get("class_name", ""))
        if class_name not in {"player", "goalkeeper", "referee"}:
            continue
        metadata = track.get("metadata", {})
        team_id = metadata.get("team_id")
        fallback = CLASS_COLORS.get(class_name, (255, 255, 255))
        if class_name == "referee":
            color = CLASS_COLORS["referee"]
        elif class_name == "player":
            color = (
                TEAM_COLORS[team_id]
                if team_id in {"TEAM_A", "TEAM_B"}
                and not metadata.get("team_uncertain", True)
                else TEAM_COLORS["UNCLASSIFIED"]
            )
        else:
            color = fallback
        draw_football_ellipse(
            output,
            track["bbox"],
            color,
            int(track["track_id"]),
        )
    _draw_ball(
        output,
        record.get("detections", []),
        record.get("possession"),
        active,
    )

    height = output.shape[0]
    legend_y = max(22, height - 18)
    cv2.putText(
        output,
        "TEAM A",
        (18, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM_COLORS["TEAM_A"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "TEAM B",
        (110, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM_COLORS["TEAM_B"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "REFEREE",
        (205, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        CLASS_COLORS["referee"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "BALL",
        (310, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        CLASS_COLORS["ball"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "UNCLASSIFIED",
        (390, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM_COLORS["UNCLASSIFIED"],
        2,
        cv2.LINE_AA,
    )
    return output


def draw_debug_frame(frame: Any, record: Mapping[str, Any]) -> Any:
    """Render diagnostic boxes, confidence, class, teams, and system status."""
    cv2 = _cv2()
    output = frame.copy()
    for track in _active_tracks(record):
        class_name = str(track.get("class_name", "unknown"))
        metadata = track.get("metadata", {})
        fallback = CLASS_COLORS.get(class_name, (255, 255, 255))
        team_id = metadata.get("team_id")
        if class_name == "referee":
            color = CLASS_COLORS["referee"]
        elif class_name == "player":
            color = (
                TEAM_COLORS[team_id]
                if team_id in {"TEAM_A", "TEAM_B"}
                and not metadata.get("team_uncertain", True)
                else TEAM_COLORS["UNCLASSIFIED"]
            )
        else:
            color = fallback
        x1, y1, x2, y2 = map(int, _coordinates(track["bbox"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        label = (
            f"ID {track['track_id']} {class_name} "
            f"conf={float(track.get('confidence', 0.0)):.2f} "
            f"team={metadata.get('team_id') or 'unclassified'} "
            f"team_conf={float(metadata.get('team_confidence', 0.0)):.2f} "
            f"uncertain={metadata.get('team_uncertain', True)} "
            f"age={track.get('age', 0)} hits={track.get('hits', 0)}"
        )
        cv2.putText(
            output,
            label,
            (x1, max(14, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        footpoint = (int((x1 + x2) / 2), y2)
        cv2.circle(output, footpoint, 3, (255, 255, 0), -1, cv2.LINE_AA)

    detections = record.get("detections", [])
    for detection in detections:
        if detection.get("class_name") != "ball":
            continue
        x1, y1, x2, y2 = map(int, _coordinates(detection["bbox"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), CLASS_COLORS["ball"], 1)
        cv2.putText(
            output,
            f"ball {float(detection.get('confidence', 0.0)):.2f}",
            (x1, max(14, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CLASS_COLORS["ball"],
            1,
            cv2.LINE_AA,
        )
    possession = record.get("possession") or {}
    possessor = possession.get("possessor_id")
    _draw_ball(output, detections, possession, _active_tracks(record))
    motion = record.get("camera_motion") or {}
    motion_text = (
        f"Camera motion: {motion.get('method')} valid, "
        f"inliers={motion.get('inlier_count', 0)}"
        if motion.get("valid")
        else f"Camera motion: invalid ({motion.get('reason', 'unknown')})"
    )
    pitch = record.get("homography") or {}
    pitch_text = (
        f"Pitch: {pitch.get('status', 'unavailable')} "
        f"quality={float(pitch.get('quality', 0.0)):.2f}"
    )
    status_lines = [
        (motion_text, (255, 255, 255)),
        (pitch_text, (0, 180, 255)),
    ]
    if possessor is not None:
        status_lines.append(
            (
                f"Possession ID {possessor} "
                f"{possession.get('possessor_team') or 'unclassified'}",
                (255, 255, 0),
            )
        )
    _draw_status_panel(output, status_lines)
    return output
