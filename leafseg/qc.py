"""Quality-control logic and overlay generation."""

from __future__ import annotations

import cv2
import numpy as np

from leafseg.config import BoundingBox, SegmentationConfig, SegmentationResult


def evaluate_qc(
    result: SegmentationResult, image_shape: tuple[int, int], config: SegmentationConfig
) -> tuple[str, str]:
    """Evaluate segmentation quality-control flags."""
    notes: list[str] = []
    height, width = image_shape[:2]

    if result.bbox is None or result.leaf_area_pixels == 0:
        notes.append("no leaf detected")
    if result.leaf_area_pixels < config.min_leaf_area_pixels:
        notes.append("leaf area too small")
    if result.mask_area_ratio < config.min_mask_area_ratio:
        notes.append("mask area ratio too small")
    if result.mask_area_ratio > config.max_mask_area_ratio:
        notes.append("mask area ratio too large")

    bbox = result.bbox
    if bbox is not None:
        touches_boundary = (
            bbox.x <= 0
            or bbox.y <= 0
            or bbox.x + bbox.width >= width
            or bbox.y + bbox.height >= height
        )
        if touches_boundary:
            notes.append("bbox touches image boundary")
        aspect = max(bbox.width / max(1, bbox.height), bbox.height / max(1, bbox.width))
        if aspect > config.max_bbox_aspect_ratio:
            notes.append("bbox aspect ratio abnormal")

    qc_flag = "review" if notes else "pass"
    return qc_flag, "; ".join(notes)


def create_qc_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: BoundingBox | None,
    result: SegmentationResult,
) -> np.ndarray:
    """Create an RGB QC overlay with contour, bbox, and text annotations."""
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(
        np.where(mask > 0, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=3)
    if bbox is not None:
        pt1 = (bbox.x, bbox.y)
        pt2 = (bbox.x + bbox.width - 1, bbox.y + bbox.height - 1)
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 255), thickness=3)

    lines = [
        result.filename,
        f"area: {result.leaf_area_pixels}",
        f"ratio: {result.mask_area_ratio:.4f}",
        f"qc: {result.qc_flag}",
    ]
    if bbox is not None:
        lines.insert(2, f"bbox: {bbox.width}x{bbox.height}")
    _draw_text_panel(overlay, lines)
    return overlay


def _draw_text_panel(image_rgb: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    margin = 10
    line_height = 24
    max_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    panel_w = min(image_rgb.shape[1], max_width + 2 * margin)
    panel_h = min(image_rgb.shape[0], line_height * len(lines) + 2 * margin)
    cv2.rectangle(image_rgb, (0, 0), (panel_w, panel_h), (255, 255, 255), thickness=-1)
    cv2.rectangle(image_rgb, (0, 0), (panel_w, panel_h), (0, 0, 0), thickness=1)
    for index, line in enumerate(lines):
        y = margin + 18 + index * line_height
        cv2.putText(image_rgb, line[:80], (margin, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
