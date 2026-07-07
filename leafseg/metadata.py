"""Metadata row construction and CSV writing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from leafseg.config import SegmentationConfig, SegmentationResult


REQUIRED_METADATA_COLUMNS = [
    "filename",
    "original_height",
    "original_width",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "leaf_area_pixels",
    "mask_area_ratio",
    "num_connected_components_before_filtering",
    "threshold_method",
    "h_min",
    "h_max",
    "s_min",
    "v_min",
    "morph_kernel_size",
    "bbox_padding",
    "qc_flag",
    "notes",
]


def result_to_metadata_row(
    result: SegmentationResult, config: SegmentationConfig, image_shape: tuple[int, int]
) -> dict[str, Any]:
    """Convert a segmentation result to a metadata CSV row."""
    height, width = image_shape[:2]
    bbox = result.bbox
    return {
        "filename": result.filename,
        "original_height": height,
        "original_width": width,
        "bbox_x": "" if bbox is None else bbox.x,
        "bbox_y": "" if bbox is None else bbox.y,
        "bbox_w": "" if bbox is None else bbox.width,
        "bbox_h": "" if bbox is None else bbox.height,
        "leaf_area_pixels": result.leaf_area_pixels,
        "mask_area_ratio": result.mask_area_ratio,
        "num_connected_components_before_filtering": result.num_connected_components_before_filtering,
        "threshold_method": config.method,
        "h_min": config.h_min,
        "h_max": config.h_max,
        "s_min": config.s_min,
        "v_min": config.v_min,
        "morph_kernel_size": config.morph_kernel_size,
        "bbox_padding": config.bbox_padding,
        "qc_flag": result.qc_flag,
        "notes": result.notes,
    }


def save_metadata(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Save batch metadata as CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = REQUIRED_METADATA_COLUMNS
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
