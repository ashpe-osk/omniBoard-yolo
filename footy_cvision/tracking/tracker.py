"""Persistent-ID tracker with aging and short occlusion recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from footy_cvision.core import Detection, Track
from footy_cvision.tracking.association import (
    AssociationConfig,
    associate,
    transform_bbox,
)


@dataclass(frozen=True)
class TrackerConfig:
    max_age: int = 10
    min_hits: int = 1
    association: AssociationConfig = AssociationConfig()

    def __post_init__(self) -> None:
        if self.max_age < 0 or self.min_hits < 1:
            raise ValueError("max_age must be non-negative and min_hits positive")


class Tracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last_frame: int | None = None
        self._last_camera_compensated_track_count = 0

    @property
    def tracks(self) -> list[Track]:
        return [self._tracks[key] for key in sorted(self._tracks)]

    @property
    def last_camera_compensated_track_count(self) -> int:
        return self._last_camera_compensated_track_count

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        camera_transform: Sequence[Sequence[float]] | None = None,
    ) -> list[Track]:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError("frame_index must increase monotonically")
        elapsed = 1 if self._last_frame is None else frame_index - self._last_frame
        for track in self._tracks.values():
            track.age += elapsed
            track.missed += elapsed

        existing = self.tracks
        self._last_camera_compensated_track_count = sum(
            camera_transform is not None and track.last_frame == self._last_frame
            for track in existing
        )
        association_tracks = [
            Track(
                track_id=track.track_id,
                bbox=(
                    transform_bbox(track.bbox, camera_transform)
                    if camera_transform is not None
                    and track.last_frame == self._last_frame
                    else track.bbox
                ),
                class_name=track.class_name,
                confidence=track.confidence,
                age=track.age,
                hits=track.hits,
                missed=track.missed,
                last_frame=track.last_frame,
                appearance=track.appearance,
                metadata=dict(track.metadata),
            )
            for track in existing
        ]
        pairs, _, unmatched_detections = associate(
            association_tracks, detections, self.config.association
        )
        for track_index, detection_index in pairs:
            track = existing[track_index]
            detection = detections[detection_index]
            track.bbox = detection.bbox
            track.class_name = detection.class_name
            track.confidence = detection.confidence
            track.appearance = detection.appearance or track.appearance
            track.metadata.update(detection.metadata)
            track.hits += 1
            track.missed = 0
            track.last_frame = frame_index

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = Track(
                track_id=self._next_id,
                bbox=detection.bbox,
                class_name=detection.class_name,
                confidence=detection.confidence,
                last_frame=frame_index,
                appearance=detection.appearance,
                metadata=dict(detection.metadata),
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.config.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]
        self._last_frame = frame_index
        return [
            track
            for track in self.tracks
            if track.missed == 0 and track.hits >= self.config.min_hits
        ]

    def snapshot(self, include_lost: bool = True) -> list[Track]:
        tracks = self.tracks
        if include_lost:
            return tracks
        return [track for track in tracks if track.missed == 0]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_frame = None
