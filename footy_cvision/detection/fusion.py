"""Class- and source-aware fusion for overlapping football detectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from footy_cvision.core import Detection


DEFAULT_CLASS_PRIORITIES: dict[str, tuple[str, ...]] = {
    "player": ("primary",),
    "goalkeeper": ("primary",),
    "referee": ("primary",),
    "ball": ("ball_referee", "primary"),
}


def canonical_class_name(class_name: str) -> str:
    normalized = class_name.strip().lower()
    aliases = {
        "players": "player",
        "goalkeepers": "goalkeeper",
        "keeper": "goalkeeper",
        "goalie": "goalkeeper",
        "referees": "referee",
        "official": "referee",
        "officials": "referee",
        "soccer ball": "ball",
        "football": "ball",
    }
    return aliases.get(normalized, normalized)


class ModelAwareFusion:
    """Select class detections by configured source priority, without blind merging.

    Detections may carry a ``model_source`` metadata field. Unconfigured
    classes are retained from all sources and left to the downstream filter.
    """

    def __init__(
        self,
        class_priorities: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        priorities = class_priorities or DEFAULT_CLASS_PRIORITIES
        self.class_priorities = {
            canonical_class_name(class_name): tuple(sources)
            for class_name, sources in priorities.items()
        }

    def __call__(
        self,
        detections_by_source: Mapping[str, Sequence[Detection]],
    ) -> list[Detection]:
        grouped: dict[str, dict[str, list[Detection]]] = {}
        for source, detections in detections_by_source.items():
            for detection in detections:
                class_name = canonical_class_name(detection.class_name)
                grouped.setdefault(class_name, {}).setdefault(source, []).append(
                    replace(
                        detection,
                        class_name=class_name,
                        metadata={
                            **detection.metadata,
                            "model_source": source,
                        },
                    )
                )

        fused: list[Detection] = []
        for class_name, by_source in grouped.items():
            priorities = self.class_priorities.get(class_name)
            if priorities is not None:
                selected = next(
                    (by_source[source] for source in priorities if source in by_source),
                    (),
                )
            else:
                selected = [
                    detection
                    for source_detections in by_source.values()
                    for detection in source_detections
                ]
            fused.extend(selected)
        return fused
