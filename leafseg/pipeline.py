"""Pipeline orchestration for one image and batches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from leafseg.config import BoundingBox, SegmentationConfig, SegmentationResult
from leafseg.io import (
    ensure_output_dirs,
    list_image_files,
    make_segmented_preview,
    read_image,
    save_mask,
    save_png_rgb,
    save_tiff,
)
from leafseg.metadata import result_to_metadata_row, save_metadata
from leafseg.morphology import clean_mask, crop_by_bbox, get_bbox
from leafseg.qc import create_qc_overlay, evaluate_qc
from leafseg.segmentation import segment_leaf

LOGGER = logging.getLogger(__name__)


def process_one_image(image_path: Path, output_dir: Path, config: SegmentationConfig) -> dict[str, Any]:
    """Process one TIFF image and return its metadata row."""
    dirs = ensure_output_dirs(output_dir)
    stem = image_path.stem
    filename = image_path.name

    try:
        original_image, image_rgb = read_image(image_path)
        raw_mask = segment_leaf(image_rgb, config)
        mask, component_count = clean_mask(raw_mask, config)
        bbox = get_bbox(mask, config.bbox_padding, image_rgb.shape[:2])
        leaf_area = int(np.count_nonzero(mask))
        mask_area_ratio = leaf_area / float(mask.size) if mask.size else 0.0
        result = SegmentationResult(
            filename=filename,
            mask=mask,
            bbox=bbox,
            leaf_area_pixels=leaf_area,
            mask_area_ratio=mask_area_ratio,
            num_connected_components_before_filtering=component_count,
            qc_flag="pending",
            notes="",
        )
        result.qc_flag, result.notes = evaluate_qc(result, image_rgb.shape[:2], config)

        save_mask(mask, dirs["masks"] / f"{stem}_leaf_mask.png")
        if bbox is not None:
            save_tiff(crop_by_bbox(original_image, bbox), dirs["crops"] / f"{stem}_leaf_crop.tif")
            save_mask(crop_by_bbox(mask, bbox), dirs["crops"] / f"{stem}_leaf_crop_mask.png")
        overlay = create_qc_overlay(image_rgb, mask, bbox, result)
        save_png_rgb(overlay, dirs["qc"] / f"{stem}_overlay.png")
        if config.save_segmented_preview:
            preview = make_segmented_preview(image_rgb, mask)
            save_png_rgb(preview, dirs["segmented_preview"] / f"{stem}_segmented_preview.png")

        return result_to_metadata_row(result, config, image_rgb.shape[:2])
    except Exception as exc:
        LOGGER.warning("Failed to process %s: %s", image_path, exc, exc_info=True)
        return _failure_row(filename, config, str(exc))


def process_folder(input_dir: Path, output_dir: Path, config: SegmentationConfig) -> None:
    """Process all TIFF files in a folder and write batch metadata."""
    ensure_output_dirs(output_dir)
    files = list_image_files(input_dir)
    LOGGER.info("Starting batch segmentation")
    LOGGER.info("Found %d supported images in %s", len(files), input_dir)

    rows: list[dict[str, Any]] = []
    counts = {"pass": 0, "review": 0, "failed": 0}
    for index, image_path in enumerate(files, start=1):
        LOGGER.info("Processing %d/%d: %s", index, len(files), image_path.name)
        row = process_one_image(image_path, output_dir, config)
        rows.append(row)
        flag = str(row.get("qc_flag", "failed"))
        counts[flag] = counts.get(flag, 0) + 1

    save_metadata(rows, output_dir / "metadata" / "segmentation_metadata.csv")
    LOGGER.info(
        "Batch complete: pass=%d review=%d failed=%d",
        counts.get("pass", 0),
        counts.get("review", 0),
        counts.get("failed", 0),
    )


def _failure_row(filename: str, config: SegmentationConfig, note: str) -> dict[str, Any]:
    result = SegmentationResult(
        filename=filename,
        mask=np.zeros((0, 0), dtype=np.uint8),
        bbox=None,
        leaf_area_pixels=0,
        mask_area_ratio=0.0,
        num_connected_components_before_filtering=0,
        qc_flag="failed",
        notes=note,
    )
    return result_to_metadata_row(result, config, (0, 0))
