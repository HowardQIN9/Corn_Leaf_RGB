"""Green-channel profile extraction along sampling lines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_green_channel(image: np.ndarray) -> np.ndarray:
    """Return the green channel from grayscale/RGB/RGBA image data."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 2:
        return arr[:, :, 1]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"Unsupported image shape for green-channel extraction: {arr.shape}")


def extract_green_profiles(
    image: np.ndarray,
    mask: np.ndarray,
    sampling_df: pd.DataFrame,
    profile_width_px: int = 5,
) -> pd.DataFrame:
    """Extract averaged green-channel profiles along sampling lines.

    For each point along each main sampling line, the green value is averaged
    across profile_width_px pixels shifted along the local tangent direction.
    """
    green = extract_green_channel(image)
    mask_binary = np.asarray(mask) > 0
    rows: list[dict[str, float | int]] = []
    for _, line in sampling_df.iterrows():
        tangent = np.array([line["tangent_x"], line["tangent_y"]], dtype=float)
        start = np.array([line["x_start"], line["y_start"]], dtype=float)
        end = np.array([line["x_end"], line["y_end"]], dtype=float)
        vector = end - start
        line_length = float(np.linalg.norm(vector))
        n_points = max(2, int(round(line_length)) + 1)
        offsets = _profile_offsets(profile_width_px)

        for position_index, fraction in enumerate(np.linspace(0.0, 1.0, n_points)):
            point = start + fraction * vector
            values: list[float] = []
            for offset in offsets:
                shifted = point + tangent * float(offset)
                x = int(round(float(shifted[0])))
                y = int(round(float(shifted[1])))
                if _inside(mask_binary, x, y):
                    values.append(float(green[y, x]))
            if not values:
                continue
            rows.append(
                {
                    "sample_id": int(line["sample_id"]),
                    "position_index": position_index,
                    "position_fraction": float(fraction),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "green_mean": float(np.mean(values)),
                    "green_min": float(np.min(values)),
                    "green_max": float(np.max(values)),
                    "n_pixels_averaged": len(values),
                    "profile_width_px": int(profile_width_px),
                }
            )
    return pd.DataFrame(rows)


def _profile_offsets(profile_width_px: int) -> list[int]:
    width = max(1, int(profile_width_px))
    half = width // 2
    if width % 2 == 1:
        return list(range(-half, half + 1))
    return list(range(-half, half))


def _inside(mask_binary: np.ndarray, x: int, y: int) -> bool:
    if y < 0 or x < 0 or y >= mask_binary.shape[0] or x >= mask_binary.shape[1]:
        return False
    return bool(mask_binary[y, x])
