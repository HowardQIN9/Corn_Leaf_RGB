"""Morphological cleanup and geometry helpers."""

from __future__ import annotations

import cv2
import numpy as np

from leafseg.config import BoundingBox, SegmentationConfig


def clean_mask(mask: np.ndarray, config: SegmentationConfig) -> tuple[np.ndarray, int]:
    """Clean a raw binary mask and return connected-component count before filtering."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_count = max(0, num_labels - 1)

    filtered = np.zeros(binary.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= config.min_leaf_area_pixels:
            filtered[labels == label] = 255

    if not np.any(filtered):
        filtered = keep_largest_component(binary)
    else:
        filtered = keep_largest_component(filtered)

    kernel_size = max(1, int(config.morph_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    filled = fill_holes(opened)
    return np.where(filled > 0, 255, 0).astype(np.uint8), component_count


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros(binary.shape, dtype=np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    flood = binary.copy()
    height, width = binary.shape[:2]
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, holes)
    return np.where(filled > 0, 255, 0).astype(np.uint8)


def get_bbox(mask: np.ndarray, padding: int, image_shape: tuple[int, int]) -> BoundingBox | None:
    """Compute a padded bounding box for foreground mask pixels."""
    binary = mask > 0
    if not np.any(binary):
        return None
    ys, xs = np.where(binary)
    height, width = image_shape[:2]
    x_min = max(0, int(xs.min()) - padding)
    y_min = max(0, int(ys.min()) - padding)
    x_max = min(width - 1, int(xs.max()) + padding)
    y_max = min(height - 1, int(ys.max()) + padding)
    return BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min + 1,
        height=y_max - y_min + 1,
    )


def crop_by_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop an image or mask using a bounding box."""
    return image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width].copy()
