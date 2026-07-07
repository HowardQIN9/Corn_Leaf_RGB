"""Configuration and result data structures for leaf segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    """User-configurable parameters for leaf segmentation."""

    method: str = "hsv"
    h_min: int = 20
    h_max: int = 95
    s_min: int = 35
    v_min: int = 25
    exg_threshold: float | None = None
    morph_kernel_size: int = 21
    min_leaf_area_pixels: int = 10000
    min_mask_area_ratio: float = 0.001
    max_mask_area_ratio: float = 0.5
    max_bbox_aspect_ratio: float = 20.0
    bbox_padding: int = 20
    save_segmented_preview: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return config values as a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class BoundingBox:
    """Pixel bounding box in x, y, width, height order."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class SegmentationResult:
    """Structured result for one segmented image."""

    filename: str
    mask: np.ndarray
    bbox: BoundingBox | None
    leaf_area_pixels: int
    mask_area_ratio: float
    num_connected_components_before_filtering: int
    qc_flag: str
    notes: str


def default_config() -> SegmentationConfig:
    """Return the default segmentation configuration."""
    return SegmentationConfig()


def load_config(path: Path) -> SegmentationConfig:
    """Load segmentation configuration from JSON or YAML."""
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError("Install PyYAML to load YAML configuration files.") from exc
            data = yaml.safe_load(handle) or {}
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    valid_names = {field.name for field in fields(SegmentationConfig)}
    unknown = set(data) - valid_names
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return SegmentationConfig(**data)


def merge_config(config: SegmentationConfig, overrides: dict[str, Any]) -> SegmentationConfig:
    """Return a new config with non-None override values applied."""
    values = config.to_dict()
    values.update({key: value for key, value in overrides.items() if value is not None})
    return SegmentationConfig(**values)
