"""Midrib-related peak detection on green-channel transverse profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclass(frozen=True)
class MidribDetectionConfig:
    """Configuration for constrained midrib-related peak detection."""

    peak_polarity: str = "bright"
    middle_min_fraction: float = 0.35
    middle_max_fraction: float = 0.70
    expected_peak_fraction: float = 0.50
    center_weight_sigma: float = 0.12
    smoothing_window_length: int = 15
    smoothing_polyorder: int = 2
    min_prominence: float = 6.0
    min_width_fraction: float = 0.01
    max_peak_position_range: float = 0.20
    min_detected_lines: int = 3


def detect_midrib_peak_for_profile(
    profile_df: pd.DataFrame,
    config: MidribDetectionConfig,
) -> dict[str, float | int | str]:
    """Detect one broad midrib-related peak for a single sampling-line profile."""
    required = {"position_fraction", "green_mean"}
    missing = required - set(profile_df.columns)
    if missing:
        raise ValueError(f"Missing required profile columns: {sorted(missing)}")

    profile = profile_df.sort_values("position_fraction").reset_index(drop=True)
    x = profile["position_fraction"].to_numpy(dtype=float)
    y = profile["green_mean"].to_numpy(dtype=float)
    sample_id = int(profile["sample_id"].iloc[0]) if "sample_id" in profile else -1
    base = {
        "sample_id": sample_id,
        "status": "no_peak",
        "peak_position_fraction": np.nan,
        "split_position_fraction": np.nan,
        "peak_left_fraction": np.nan,
        "peak_right_fraction": np.nan,
        "peak_prominence": np.nan,
        "peak_width_fraction": np.nan,
        "peak_score": np.nan,
        "notes": "",
    }
    if len(profile) < 5:
        base["notes"] = "too few profile points"
        return base

    y_smooth = _smooth_profile(y, config.smoothing_window_length, config.smoothing_polyorder)
    if config.peak_polarity == "dark":
        detection_signal = -y_smooth
    elif config.peak_polarity == "bright":
        detection_signal = y_smooth
    else:
        raise ValueError(f"Unsupported peak_polarity: {config.peak_polarity}")
    middle_mask = (x >= config.middle_min_fraction) & (x <= config.middle_max_fraction)
    if int(np.count_nonzero(middle_mask)) < 5:
        base["notes"] = "too few points in middle search region"
        return base

    middle_indices = np.flatnonzero(middle_mask)
    y_middle = detection_signal[middle_indices]
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    min_width_samples = max(1, int(round(config.min_width_fraction / max(dx, 1e-9))))
    peaks, properties = find_peaks(y_middle, prominence=config.min_prominence, width=min_width_samples)
    if len(peaks) == 0:
        base["notes"] = "no broad prominent peak in middle region"
        return base

    widths = peak_widths(y_middle, peaks, rel_height=0.5)
    width_samples = widths[0]
    prominences = properties["prominences"]
    middle_x = x[middle_indices]
    candidate_positions = middle_x[peaks]
    center_weights = np.exp(
        -0.5 * ((candidate_positions - config.expected_peak_fraction) / config.center_weight_sigma) ** 2
    )
    scores = prominences * width_samples * center_weights
    best_local = int(np.argmax(scores))
    local_peak_index = int(peaks[best_local])
    global_peak_index = int(middle_indices[local_peak_index])
    width_fraction = float(width_samples[best_local] * dx)
    if width_fraction < config.min_width_fraction:
        base["notes"] = "best peak is narrower than minimum width"
        return base

    left_ip = float(widths[2][best_local])
    right_ip = float(widths[3][best_local])
    peak_left_fraction = float(np.interp(left_ip, np.arange(len(middle_x)), middle_x))
    peak_right_fraction = float(np.interp(right_ip, np.arange(len(middle_x)), middle_x))
    split_position_fraction = float(x[global_peak_index])

    return {
        "sample_id": sample_id,
        "status": "detected",
        "peak_polarity": config.peak_polarity,
        "peak_position_fraction": float(x[global_peak_index]),
        "split_position_fraction": split_position_fraction,
        "peak_left_fraction": peak_left_fraction,
        "peak_right_fraction": peak_right_fraction,
        "peak_prominence": float(prominences[best_local]),
        "peak_width_fraction": width_fraction,
        "peak_score": float(scores[best_local]),
        "notes": "",
    }


def detect_midrib_peaks(
    profiles_df: pd.DataFrame,
    config: MidribDetectionConfig,
) -> pd.DataFrame:
    """Detect constrained midrib-related peaks for every profile line."""
    group_cols = ["source_profile_file", "sample_id"]
    rows: list[dict[str, float | int | str]] = []
    for keys, profile in profiles_df.groupby(group_cols, sort=True):
        source_profile_file, sample_id = keys
        result = detect_midrib_peak_for_profile(profile, config)
        first = profile.iloc[0]
        rows.append(
            {
                "prefix": first.get("prefix", ""),
                "plot_number": first.get("plot_number", ""),
                "geno": first.get("geno", ""),
                "plant_number": first.get("plant_number", ""),
                "leaf": first.get("leaf", ""),
                "timestamp": first.get("timestamp", ""),
                "source_profile_file": source_profile_file,
                "filename": first.get("filename", ""),
                "peak_polarity": config.peak_polarity,
                **result,
            }
        )
    return pd.DataFrame(rows)


def summarize_midrib_detection(
    line_results: pd.DataFrame,
    config: MidribDetectionConfig,
) -> pd.DataFrame:
    """Summarize 5-line midrib detection consistency for each leaf."""
    rows: list[dict[str, float | int | str]] = []
    for source_profile_file, group in line_results.groupby("source_profile_file", sort=True):
        first = group.iloc[0]
        detected = group[group["status"] == "detected"].copy()
        n_detected = len(detected)
        notes: list[str] = []
        qc_flag = "pass"
        peak_range = np.nan
        median_split = np.nan
        median_peak = np.nan
        if n_detected < config.min_detected_lines:
            qc_flag = "review"
            notes.append("too few detected midrib peaks")
        else:
            peak_positions = detected["peak_position_fraction"].astype(float)
            split_positions = detected["split_position_fraction"].astype(float)
            peak_range = float(peak_positions.max() - peak_positions.min())
            median_peak = float(peak_positions.median())
            median_split = float(split_positions.median())
            if peak_range > config.max_peak_position_range:
                qc_flag = "review"
                notes.append("detected peak positions are inconsistent")

        rows.append(
            {
                "prefix": first.get("prefix", ""),
                "plot_number": first.get("plot_number", ""),
                "geno": first.get("geno", ""),
                "plant_number": first.get("plant_number", ""),
                "leaf": first.get("leaf", ""),
                "timestamp": first.get("timestamp", ""),
                "source_profile_file": source_profile_file,
                "n_profile_lines": len(group),
                "n_detected_peaks": n_detected,
                "median_peak_position_fraction": median_peak,
                "median_split_position_fraction": median_split,
                "peak_position_range": peak_range,
                "qc_flag": qc_flag,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows)


def annotate_profile_sides(profiles_df: pd.DataFrame, line_results: pd.DataFrame) -> pd.DataFrame:
    """Add side-of-midrib columns to long-format profile data."""
    merge_cols = [
        "source_profile_file",
        "sample_id",
        "status",
        "peak_position_fraction",
        "split_position_fraction",
        "peak_left_fraction",
        "peak_right_fraction",
    ]
    annotated = profiles_df.merge(line_results[merge_cols], on=["source_profile_file", "sample_id"], how="left")
    position = annotated["position_fraction"].astype(float)
    split = annotated["split_position_fraction"].astype(float)
    detected = annotated["status"] == "detected"
    annotated["side_of_midrib"] = "unknown"
    annotated.loc[detected & (position < split), "side_of_midrib"] = "upper"
    annotated.loc[detected & (position > split), "side_of_midrib"] = "lower"
    annotated.loc[detected & np.isclose(position, split), "side_of_midrib"] = "midrib_center"
    annotated["distance_from_midrib_split"] = position - split
    annotated["is_midrib_peak_region"] = (
        detected
        & (position >= annotated["peak_left_fraction"].astype(float))
        & (position <= annotated["peak_right_fraction"].astype(float))
    )
    return annotated


def _smooth_profile(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    n_points = len(values)
    if n_points <= polyorder + 1:
        return values.astype(float)
    window = min(int(window_length), n_points)
    if window % 2 == 0:
        window -= 1
    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window:
        return values.astype(float)
    return savgol_filter(values.astype(float), window_length=window, polyorder=polyorder, mode="interp")
