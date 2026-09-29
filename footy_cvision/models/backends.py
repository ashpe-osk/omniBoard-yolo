"""Lazy adapters for optional Ultralytics and Roboflow inference packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from os import PathLike
from typing import Any

from footy_cvision.core import Detection
from footy_cvision.detection.filter import bbox_from


def normalize_detections(
    results: Any,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
) -> list[Detection]:
    """Normalize common Ultralytics/Roboflow outputs to package detections."""
    if results is None:
        return []
    if isinstance(results, Mapping):
        if "predictions" in results:
            results = results["predictions"]
            if results is None:
                return []
        elif any(key in results for key in ("bbox", "xyxy", "x")):
            results = [results]
        else:
            raise ValueError("Unrecognized detection mapping")
    elif not isinstance(results, Sequence) and hasattr(results, "boxes"):
        results = [results]
    elif not isinstance(results, Sequence):
        raise TypeError("Inference results must be a mapping or sequence")

    normalized: list[Detection] = []
    for result in results:
        if isinstance(result, Detection):
            normalized.append(result)
            continue
        if hasattr(result, "boxes"):
            normalized.extend(_ultralytics_result(result, class_names))
            continue
        if not isinstance(result, Mapping):
            raise TypeError(f"Unsupported detection result: {type(result).__name__}")
        if "bbox" in result:
            box = bbox_from(result["bbox"])
        elif "xyxy" in result:
            box = bbox_from(_as_list(result["xyxy"]))
        elif all(key in result for key in ("x", "y", "width", "height")):
            center_x = float(result["x"])
            center_y = float(result["y"])
            width = float(result["width"])
            height = float(result["height"])
            box = bbox_from(
                (
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                )
            )
        else:
            raise ValueError("Detection mapping needs bbox, xyxy, or x/y/width/height")
        confidence = result.get("confidence", result.get("conf"))
        if confidence is None:
            raise ValueError("Detection mapping is missing confidence")
        class_name = result.get("class_name", result.get("class"))
        if class_name is None and "cls" in result:
            class_id = int(_scalar(result["cls"]))
            class_name = _class_name(class_names or {}, class_id)
        if class_name is None:
            raise ValueError("Detection mapping is missing class_name/class/cls")
        metadata = dict(result.get("metadata", {}))
        normalized.append(
            Detection(
                box,
                float(_scalar(confidence)),
                str(class_name),
                appearance=(
                    tuple(map(float, result["appearance"]))
                    if result.get("appearance") is not None
                    else None
                ),
                metadata=metadata,
            )
        )
    return normalized


class UltralyticsBackend:
    """Callable detector adapter for an Ultralytics model or weights path."""

    def __init__(
        self,
        model: Any,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
        **predict_kwargs: Any,
    ) -> None:
        if isinstance(model, (str, bytes, PathLike)):
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "UltralyticsBackend needs the optional 'ultralytics' package"
                ) from error
            self.model = YOLO(model)
        else:
            self.model = model
        self.class_names = class_names
        self.predict_kwargs = predict_kwargs

    def __call__(self, frame: Any) -> list[Detection]:
        results = self.model.predict(frame, **self.predict_kwargs)
        names = self.class_names
        if names is None:
            names = getattr(self.model, "names", None)
        return normalize_detections(results, names)


class RoboflowBackend:
    """Callable detector adapter for an inference-sdk Roboflow model."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        api_url: str = "https://detect.roboflow.com",
    ) -> None:
        if not api_key:
            raise ValueError("A Roboflow API key is required")
        if not model_id:
            raise ValueError("A Roboflow model_id is required")
        try:
            client_module = import_module("inference_sdk")
            inference_client = client_module.InferenceHTTPClient
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "RoboflowBackend needs the optional 'inference-sdk' package"
            ) from error
        self.model_id = model_id
        self.client = inference_client(api_url=api_url, api_key=api_key)

    def __call__(self, frame: Any) -> list[Detection]:
        results = self.client.infer(frame, model_id=self.model_id)
        return normalize_detections(results)


def _ultralytics_result(
    result: Any, class_names: Mapping[int, str] | Sequence[str] | None
) -> list[Detection]:
    boxes = result.boxes
    if boxes is None:
        return []
    coordinates = _as_list(boxes.xyxy)
    confidences = _as_list(boxes.conf)
    class_ids = _as_list(boxes.cls)
    result_names = class_names or getattr(result, "names", None) or {}
    detections: list[Detection] = []
    for box, confidence, class_id in zip(coordinates, confidences, class_ids):
        identifier = int(_scalar(class_id))
        detections.append(
            Detection(
                bbox_from(box),
                float(_scalar(confidence)),
                str(_class_name(result_names, identifier)),
            )
        )
    return detections


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _class_name(names: Mapping[int, str] | Sequence[str], identifier: int) -> str | int:
    if isinstance(names, Mapping):
        return names.get(identifier, identifier)
    if 0 <= identifier < len(names):
        return names[identifier]
    return identifier
