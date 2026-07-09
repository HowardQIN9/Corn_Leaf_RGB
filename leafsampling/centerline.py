"""Geometric centerline extraction from binary leaf masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_leaf_boundaries(mask: np.ndarray, min_leaf_width: int) -> pd.DataFrame:
    """Compute top, bottom, width, and raw geometric center for each valid x column."""
    mask_binary = np.asarray(mask) > 0
    rows: list[dict[str, float | int]] = []
    for x in range(mask_binary.shape[1]):
        y_values = np.flatnonzero(mask_binary[:, x])
        if y_values.size == 0:
            continue
        y_top = int(y_values.min())
        y_bottom = int(y_values.max())
        leaf_width = y_bottom - y_top + 1
        if leaf_width < min_leaf_width:
            continue
        rows.append(
            {
                "point_id": len(rows),
                "x": float(x),
                "y_top": y_top,
                "y_bottom": y_bottom,
                "leaf_width": leaf_width,
                "y_center_raw": (y_top + y_bottom) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def trim_centerline_points(df: pd.DataFrame, trim_ratio: float) -> pd.DataFrame:
    """Trim extreme leaf-tip centerline points from both ends."""
    if df.empty or trim_ratio <= 0:
        return df.reset_index(drop=True).assign(point_id=lambda data: np.arange(len(data)))
    n_points = len(df)
    trim_count = int(np.floor(n_points * trim_ratio))
    if trim_count == 0:
        return df.reset_index(drop=True).assign(point_id=lambda data: np.arange(len(data)))
    if 2 * trim_count >= n_points:
        return df.iloc[0:0].copy()
    trimmed = df.iloc[trim_count : n_points - trim_count].copy()
    trimmed["point_id"] = np.arange(len(trimmed))
    return trimmed.reset_index(drop=True)


def smooth_centerline(df: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    """Smooth raw geometric centerline y positions with a Savitzky-Golay filter."""
    if df.empty:
        raise ValueError("Cannot smooth an empty centerline.")
    output = df.copy()
    n_points = len(output)
    effective_window = _effective_savgol_window(n_points, window_length, polyorder)
    if effective_window is None:
        output["y_center_smooth"] = output["y_center_raw"].astype(float)
        output.attrs["smoothing_window_length"] = 1
        return output
    output["y_center_smooth"] = savgol_filter(
        output["y_center_raw"].to_numpy(dtype=float),
        window_length=effective_window,
        polyorder=polyorder,
        mode="interp",
    )
    output.attrs["smoothing_window_length"] = effective_window
    return output


def compute_horizontal_centerline(df: pd.DataFrame) -> pd.DataFrame:
    """Replace local centerline curvature with one robust horizontal centerline."""
    if df.empty:
        raise ValueError("Cannot compute a horizontal centerline from empty data.")
    output = df.copy()
    y_center = float(np.median(output["y_center_raw"].to_numpy(dtype=float)))
    output["y_center_smooth"] = y_center
    output.attrs["smoothing_window_length"] = 1
    return output


def compute_tangent_normal(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized local tangent and normal vectors along the centerline."""
    if df.empty:
        raise ValueError("Cannot compute geometry for an empty centerline.")
    output = df.copy()
    x = output["x"].to_numpy(dtype=float)
    y = output["y_center_smooth"].to_numpy(dtype=float)
    if len(output) == 1:
        dy_dx = np.array([0.0])
    else:
        dy_dx = np.gradient(y, x)

    tangent = np.column_stack([np.ones_like(dy_dx), dy_dx])
    tangent_norm = np.linalg.norm(tangent, axis=1)
    tangent = tangent / tangent_norm[:, None]

    normal = np.column_stack([-dy_dx, np.ones_like(dy_dx)])
    normal_norm = np.linalg.norm(normal, axis=1)
    normal = normal / normal_norm[:, None]

    output["dy_dx"] = dy_dx
    output["tangent_x"] = tangent[:, 0]
    output["tangent_y"] = tangent[:, 1]
    output["normal_x"] = normal[:, 0]
    output["normal_y"] = normal[:, 1]
    return output


def _effective_savgol_window(n_points: int, requested_window: int, polyorder: int) -> int | None:
    if n_points <= polyorder + 1:
        return None
    window = min(int(requested_window), n_points)
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return None
    return max(3, window)
