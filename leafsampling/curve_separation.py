"""Separate split green profiles into baseline and peak components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def create_valley_envelope(
    values: np.ndarray,
    valley_distance: int = 7,
    smooth_window: int = 21,
) -> np.ndarray:
    """Create a smooth lower envelope through local valleys and endpoints."""
    y_raw = _fill_missing(np.asarray(values, dtype=float))
    n_points = y_raw.size
    if n_points == 0:
        return y_raw
    if n_points < 3:
        return np.full_like(y_raw, float(np.nanmean(y_raw)))

    distance = max(1, min(int(valley_distance), n_points))
    valleys, _ = find_peaks(-y_raw, distance=distance)
    x_coords = np.arange(n_points)
    valley_x = np.concatenate(([0], valleys, [n_points - 1]))
    valley_y = y_raw[valley_x]

    unique_x, indices = np.unique(valley_x, return_index=True)
    valley_x = unique_x
    valley_y = valley_y[indices]
    if len(valley_x) < 2:
        return np.full_like(y_raw, float(np.nanmin(y_raw)))

    jagged_envelope = np.interp(x_coords, valley_x, valley_y)
    window = _valid_savgol_window(n_points, smooth_window, polyorder=2)
    if window is None:
        return jagged_envelope
    return savgol_filter(jagged_envelope, window_length=window, polyorder=2, mode="interp")


def separate_profile_components(
    split_profiles: pd.DataFrame,
    *,
    value_column: str = "green_mean",
    valley_distance: int = 7,
    smooth_window: int = 21,
) -> pd.DataFrame:
    """Add lower-envelope baseline and residual peak columns to split profiles."""
    required = {"source_profile_file", "sample_id", "midrib_side", "distance_index", value_column}
    missing = required - set(split_profiles.columns)
    if missing:
        raise ValueError(f"Missing split profile columns: {sorted(missing)}")

    output_groups: list[pd.DataFrame] = []
    group_cols = ["source_profile_file", "sample_id", "midrib_side"]
    for _, group in split_profiles.groupby(group_cols, sort=True):
        ordered = group.sort_values("distance_index").copy()
        y_raw = ordered[value_column].to_numpy(dtype=float)
        baseline = create_valley_envelope(
            y_raw,
            valley_distance=valley_distance,
            smooth_window=smooth_window,
        )
        ordered[f"{value_column}_meso"] = baseline
        ordered[f"{value_column}_peak"] = y_raw - baseline
        output_groups.append(ordered)

    if not output_groups:
        return split_profiles.copy()
    separated = pd.concat(output_groups, ignore_index=True)
    return separated.sort_values(group_cols + ["distance_index"]).reset_index(drop=True)


def _fill_missing(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)
    series = pd.Series(values, dtype="float64")
    if series.isna().all():
        return np.zeros(values.size, dtype=float)
    return series.interpolate(limit_direction="both").to_numpy(dtype=float)


def _valid_savgol_window(n_points: int, requested_window: int, polyorder: int) -> int | None:
    window = min(int(requested_window), n_points)
    if window % 2 == 0:
        window -= 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        return None
    return window
