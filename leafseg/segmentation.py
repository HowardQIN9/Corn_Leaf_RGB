"""Leaf segmentation algorithms."""

from __future__ import annotations

import cv2
import numpy as np

from leafseg.config import SegmentationConfig


def segment_leaf_hsv(image_rgb: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Segment green leaf pixels using OpenCV HSV thresholds."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([config.h_min, config.s_min, config.v_min], dtype=np.uint8)
    upper = np.array([config.h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return _as_binary_uint8(mask)


def segment_leaf_exg(image_rgb: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Segment leaf pixels using the excess green index."""
    rgb = image_rgb.astype(np.float32) / 255.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    exg = 2.0 * green - red - blue
    if config.exg_threshold is None:
        threshold_value, mask = cv2.threshold(
            _normalize_float_to_uint8(exg), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _ = threshold_value
        return _as_binary_uint8(mask)
    mask = (exg >= config.exg_threshold).astype(np.uint8) * 255
    return _as_binary_uint8(mask)


def segment_leaf(image_rgb: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Segment a leaf using the configured method."""
    method = config.method.lower()
    if method == "hsv":
        return segment_leaf_hsv(image_rgb, config)
    if method == "exg":
        return segment_leaf_exg(image_rgb, config)
    if method in {"hsv_exg_fallback", "hsv+exg"}:
        hsv_mask = segment_leaf_hsv(image_rgb, config)
        area_ratio = float(np.count_nonzero(hsv_mask)) / float(hsv_mask.size)
        if area_ratio < config.min_mask_area_ratio:
            return segment_leaf_exg(image_rgb, config)
        return hsv_mask
    raise ValueError(f"Unsupported segmentation method: {config.method}")


def _normalize_float_to_uint8(image: np.ndarray) -> np.ndarray:
    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - min_value) / (max_value - min_value)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _as_binary_uint8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)
