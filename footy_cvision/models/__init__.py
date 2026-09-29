"""Optional model inference adapters."""

from .backends import RoboflowBackend, UltralyticsBackend, normalize_detections

__all__ = ["RoboflowBackend", "UltralyticsBackend", "normalize_detections"]
