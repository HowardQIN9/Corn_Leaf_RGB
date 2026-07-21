"""Feature extraction for split leaf intensity profiles.

The feature families follow the derivative and curvature workflow used in the
previous slope analysis while adapting it to the current long-format curves.
Each half-profile is resampled on a common midrib-to-edge distance grid before
features are calculated so profiles with different pixel lengths are
comparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


GROUP_COLUMNS = ("source_profile_file", "sample_id", "midrib_side")
DEFAULT_VALUE_COLUMNS = ("green_mean", "green_mean_meso", "green_mean_peak")
DEFAULT_METADATA_COLUMNS = (
    "prefix",
    "plot_number",
    "geno",
    "plant_number",
    "leaf",
    "timestamp",
    "source_profile_file",
    "filename",
)

SIGNAL_FEATURE_NAMES = (
    "n_points",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "range",
    "mean_abs",
    "max_abs",
    "pos_frac",
    "neg_frac",
    "zero_cross_rate",
    "energy",
    "energy_density",
    "rms",
    "q25",
    "q75",
    "iqr",
    "abs_q95",
    "pos_sum",
    "neg_abs_sum",
    "pos_neg_ratio",
    "roughness",
    "roughness_std",
    "high_abs_frac_p90",
    "high_abs_mean_p90",
    "auc",
    "abs_auc",
    "linear_slope",
)

CURVATURE_FEATURE_NAMES = (
    "n_points",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "range",
    "q25",
    "q75",
    "iqr",
    "q95",
    "energy",
    "energy_density",
    "rms",
)

PEAK_FEATURE_NAMES = (
    "count",
    "density",
    "height_mean",
    "height_max",
    "prominence_mean",
    "prominence_max",
    "width_mean",
    "width_max",
)

CURVE_DESCRIPTOR_COLUMNS = (
    "curve_n_original_points",
    "curve_distance_min",
    "curve_distance_max",
    "curve_distance_coverage",
)


def zero_crossing_rate(values: Sequence[float] | np.ndarray) -> float:
    """Return sign-change frequency while ignoring exact zero samples."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return np.nan

    signs = np.sign(y)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0.0
    crossings = np.sum(nonzero[1:] != nonzero[:-1])
    return float(crossings / max(y.size - 1, 1))


def signal_features(
    values: Sequence[float] | np.ndarray,
    prefix: str,
    *,
    x: Sequence[float] | np.ndarray | None = None,
    min_points: int = 5,
) -> dict[str, float]:
    """Calculate distribution, energy, balance, and roughness features."""
    y = np.asarray(values, dtype=float)
    if x is None:
        x_values = np.arange(y.size, dtype=float)
    else:
        x_values = np.asarray(x, dtype=float)
        if x_values.size != y.size:
            raise ValueError("x and values must have the same length")

    valid = np.isfinite(y) & np.isfinite(x_values)
    y = y[valid]
    x_values = x_values[valid]
    features = _empty_features(prefix, SIGNAL_FEATURE_NAMES)
    features[f"{prefix}_n_points"] = float(y.size)
    if y.size < min_points:
        return features

    order = np.argsort(x_values)
    x_values = x_values[order]
    y = y[order]
    abs_y = np.abs(y)
    q25, q75 = np.percentile(y, [25, 75])
    pos_sum = float(np.sum(y[y > 0]))
    neg_abs_sum = float(np.sum(np.abs(y[y < 0])))

    features.update(
        {
            f"{prefix}_mean": float(np.mean(y)),
            f"{prefix}_median": float(np.median(y)),
            f"{prefix}_std": float(np.std(y, ddof=0)),
            f"{prefix}_var": float(np.var(y, ddof=0)),
            f"{prefix}_min": float(np.min(y)),
            f"{prefix}_max": float(np.max(y)),
            f"{prefix}_range": float(np.max(y) - np.min(y)),
            f"{prefix}_mean_abs": float(np.mean(abs_y)),
            f"{prefix}_max_abs": float(np.max(abs_y)),
            f"{prefix}_pos_frac": float(np.mean(y > 0)),
            f"{prefix}_neg_frac": float(np.mean(y < 0)),
            f"{prefix}_zero_cross_rate": zero_crossing_rate(y),
            f"{prefix}_energy": float(np.sum(y**2)),
            f"{prefix}_energy_density": float(np.mean(y**2)),
            f"{prefix}_rms": float(np.sqrt(np.mean(y**2))),
            f"{prefix}_q25": float(q25),
            f"{prefix}_q75": float(q75),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_abs_q95": float(np.percentile(abs_y, 95)),
            f"{prefix}_pos_sum": pos_sum,
            f"{prefix}_neg_abs_sum": neg_abs_sum,
            f"{prefix}_pos_neg_ratio": float(pos_sum / (neg_abs_sum + 1e-8)),
            f"{prefix}_auc": float(np.trapezoid(y, x_values)),
            f"{prefix}_abs_auc": float(np.trapezoid(abs_y, x_values)),
            f"{prefix}_linear_slope": float(np.polyfit(x_values, y, 1)[0]),
        }
    )

    differences = np.diff(y)
    features[f"{prefix}_roughness"] = float(np.mean(np.abs(differences)))
    features[f"{prefix}_roughness_std"] = float(np.std(differences, ddof=0))

    high_threshold = float(np.percentile(abs_y, 90))
    high_mask = abs_y >= high_threshold
    features[f"{prefix}_high_abs_frac_p90"] = float(np.mean(high_mask))
    features[f"{prefix}_high_abs_mean_p90"] = float(np.mean(abs_y[high_mask]))
    return features


def curvature_features(
    first_derivative: Sequence[float] | np.ndarray,
    second_derivative: Sequence[float] | np.ndarray,
    prefix: str,
    *,
    min_points: int = 5,
) -> dict[str, float]:
    """Calculate geometric curvature features from aligned derivatives."""
    d1 = np.asarray(first_derivative, dtype=float)
    d2 = np.asarray(second_derivative, dtype=float)
    n_points = min(d1.size, d2.size)
    d1 = d1[:n_points]
    d2 = d2[:n_points]
    valid = np.isfinite(d1) & np.isfinite(d2)
    d1 = d1[valid]
    d2 = d2[valid]

    features = _empty_features(prefix, CURVATURE_FEATURE_NAMES)
    features[f"{prefix}_n_points"] = float(d1.size)
    if d1.size < min_points:
        return features

    curvature = np.abs(d2) / np.power(1.0 + d1**2, 1.5)
    q25, q75 = np.percentile(curvature, [25, 75])
    features.update(
        {
            f"{prefix}_mean": float(np.mean(curvature)),
            f"{prefix}_median": float(np.median(curvature)),
            f"{prefix}_std": float(np.std(curvature, ddof=0)),
            f"{prefix}_var": float(np.var(curvature, ddof=0)),
            f"{prefix}_min": float(np.min(curvature)),
            f"{prefix}_max": float(np.max(curvature)),
            f"{prefix}_range": float(np.max(curvature) - np.min(curvature)),
            f"{prefix}_q25": float(q25),
            f"{prefix}_q75": float(q75),
            f"{prefix}_iqr": float(q75 - q25),
            f"{prefix}_q95": float(np.percentile(curvature, 95)),
            f"{prefix}_energy": float(np.sum(curvature**2)),
            f"{prefix}_energy_density": float(np.mean(curvature**2)),
            f"{prefix}_rms": float(np.sqrt(np.mean(curvature**2))),
        }
    )
    return features


def peak_features(
    values: Sequence[float] | np.ndarray,
    prefix: str,
    *,
    x: Sequence[float] | np.ndarray | None = None,
    min_points: int = 5,
    prominence_fraction: float = 0.10,
) -> dict[str, float]:
    """Summarize positive local peaks using a range-scaled prominence."""
    y = np.asarray(values, dtype=float)
    if x is None:
        x_values = np.arange(y.size, dtype=float)
    else:
        x_values = np.asarray(x, dtype=float)
        if x_values.size != y.size:
            raise ValueError("x and values must have the same length")

    valid = np.isfinite(y) & np.isfinite(x_values)
    y = y[valid]
    x_values = x_values[valid]
    features = _empty_features(prefix, PEAK_FEATURE_NAMES)
    if y.size < min_points:
        return features

    robust_range = float(np.percentile(y, 95) - np.percentile(y, 5))
    prominence = max(robust_range * float(prominence_fraction), np.finfo(float).eps)
    peaks, properties = find_peaks(y, prominence=prominence)
    features[f"{prefix}_count"] = float(peaks.size)
    features[f"{prefix}_density"] = float(peaks.size / y.size)
    if peaks.size == 0:
        return features

    widths = peak_widths(y, peaks, rel_height=0.5)[0]
    spacing = float(np.median(np.diff(x_values))) if x_values.size >= 2 else 1.0
    widths = widths * spacing
    prominences = properties["prominences"]
    heights = y[peaks]
    features.update(
        {
            f"{prefix}_height_mean": float(np.mean(heights)),
            f"{prefix}_height_max": float(np.max(heights)),
            f"{prefix}_prominence_mean": float(np.mean(prominences)),
            f"{prefix}_prominence_max": float(np.max(prominences)),
            f"{prefix}_width_mean": float(np.mean(widths)),
            f"{prefix}_width_max": float(np.max(widths)),
        }
    )
    return features


def extract_curve_features(
    profiles: pd.DataFrame,
    *,
    value_columns: Sequence[str] = DEFAULT_VALUE_COLUMNS,
    derivative_column: str | None = "green_mean_meso",
    distance_column: str = "relative_distance_from_midrib",
    n_zones: int = 3,
    resample_points: int = 256,
    min_points: int = 5,
    peak_prominence_fraction: float = 0.10,
) -> pd.DataFrame:
    """Return one feature row per sample line and midrib side.

    Features are calculated for the full half-profile and for ``n_zones``
    equally sized midrib-to-edge zones. Derivative and curvature features are
    calculated only for ``derivative_column`` to mirror the previous
    mesophyll-slope workflow without duplicating thousands of correlated
    derivative features for every signal component.
    """
    value_columns = tuple(dict.fromkeys(value_columns))
    if not value_columns:
        raise ValueError("At least one value column is required")
    resample_columns = value_columns
    if derivative_column is not None and derivative_column not in resample_columns:
        resample_columns = (*resample_columns, derivative_column)
    required = {*GROUP_COLUMNS, distance_column, *value_columns}
    if derivative_column is not None:
        required.add(derivative_column)
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing feature input columns: {sorted(missing)}")
    if n_zones < 1:
        raise ValueError("n_zones must be at least 1")
    if resample_points < 3:
        raise ValueError("resample_points must be at least 3")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")
    if not 0 <= peak_prominence_fraction:
        raise ValueError("peak_prominence_fraction must be non-negative")

    rows: list[dict[str, object]] = []
    metadata_columns = [column for column in DEFAULT_METADATA_COLUMNS if column in profiles.columns]
    for group_key, group in profiles.groupby(list(GROUP_COLUMNS), sort=True, dropna=False):
        ordered = group.sort_values(distance_column)
        x_grid, resampled, distance_stats = _resample_group(
            ordered,
            value_columns=resample_columns,
            distance_column=distance_column,
            resample_points=resample_points,
        )
        row: dict[str, object] = dict(zip(GROUP_COLUMNS, group_key, strict=True))
        first = ordered.iloc[0]
        for column in metadata_columns:
            row[column] = first[column]
        row.update(distance_stats)

        derivative_1: np.ndarray | None = None
        derivative_2: np.ndarray | None = None
        if derivative_column is not None:
            derivative_1, derivative_2 = _calculate_derivatives(x_grid, resampled[derivative_column])

        for zone_name, lower, upper in _zone_definitions(n_zones):
            zone_mask = (x_grid >= lower) & (x_grid <= upper if upper == 1.0 else x_grid < upper)
            zone_x = x_grid[zone_mask]
            for value_column in value_columns:
                zone_values = resampled[value_column][zone_mask]
                signal_prefix = f"{value_column}__{zone_name}__signal"
                row.update(signal_features(zone_values, signal_prefix, x=zone_x, min_points=min_points))
                if value_column.endswith("_peak"):
                    peak_prefix = f"{value_column}__{zone_name}__peaks"
                    row.update(
                        peak_features(
                            zone_values,
                            peak_prefix,
                            x=zone_x,
                            min_points=min_points,
                            prominence_fraction=peak_prominence_fraction,
                        )
                    )

            if derivative_column is not None and derivative_1 is not None and derivative_2 is not None:
                d1_zone = derivative_1[zone_mask]
                d2_zone = derivative_2[zone_mask]
                row.update(
                    signal_features(
                        d1_zone,
                        f"{derivative_column}__{zone_name}__d1",
                        x=zone_x,
                        min_points=min_points,
                    )
                )
                row.update(
                    signal_features(
                        d2_zone,
                        f"{derivative_column}__{zone_name}__d2",
                        x=zone_x,
                        min_points=min_points,
                    )
                )
                row.update(
                    curvature_features(
                        d1_zone,
                        d2_zone,
                        f"{derivative_column}__{zone_name}__curvature",
                        min_points=min_points,
                    )
                )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(GROUP_COLUMNS))
    result = pd.DataFrame(rows)
    identifier_columns = _ordered_existing_columns(result, (*DEFAULT_METADATA_COLUMNS, "sample_id", "midrib_side"))
    descriptor_columns = _ordered_existing_columns(result, CURVE_DESCRIPTOR_COLUMNS)
    feature_columns = sorted(set(result.columns) - set(identifier_columns) - set(descriptor_columns))
    return result[identifier_columns + descriptor_columns + feature_columns]


def aggregate_image_features(
    curve_features: pd.DataFrame,
    *,
    aggregations: Sequence[str] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Aggregate half-curve features into one row per source image."""
    if "source_profile_file" not in curve_features.columns:
        raise ValueError("curve feature table is missing source_profile_file")
    supported = {"mean", "std", "min", "max", "median"}
    unsupported = set(aggregations) - supported
    if unsupported:
        raise ValueError(f"Unsupported aggregations: {sorted(unsupported)}")
    if not aggregations:
        raise ValueError("At least one aggregation is required")

    feature_columns = [
        column
        for column in curve_features.columns
        if column in CURVE_DESCRIPTOR_COLUMNS or "__" in str(column)
    ]
    if not feature_columns:
        raise ValueError("No numeric curve feature columns were found")

    group = curve_features.groupby("source_profile_file", sort=True, dropna=False)
    aggregated = group[feature_columns].agg(list(aggregations))
    aggregated.columns = [
        f"all_curves__{feature_name}__{aggregation}"
        for feature_name, aggregation in aggregated.columns.to_flat_index()
    ]

    metadata_columns = [
        column
        for column in DEFAULT_METADATA_COLUMNS
        if column in curve_features.columns and column != "source_profile_file"
    ]
    metadata = group[metadata_columns].first() if metadata_columns else pd.DataFrame(index=aggregated.index)
    n_curves = group.size().rename("n_curves")
    output = pd.concat([metadata, n_curves, aggregated], axis=1).reset_index()
    leading = _ordered_existing_columns(output, DEFAULT_METADATA_COLUMNS) + ["n_curves"]
    remaining = [column for column in output.columns if column not in leading]
    return output[leading + remaining]


def _resample_group(
    group: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    distance_column: str,
    resample_points: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    numeric = group[[distance_column, *value_columns]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric[np.isfinite(numeric[distance_column])]
    numeric = numeric[(numeric[distance_column] >= 0.0) & (numeric[distance_column] <= 1.0)]
    numeric = numeric.groupby(distance_column, as_index=False).mean(numeric_only=True).sort_values(distance_column)
    x_grid = np.linspace(0.0, 1.0, resample_points)
    resampled = {column: np.full(resample_points, np.nan, dtype=float) for column in value_columns}

    if numeric.empty:
        stats = {
            "curve_n_original_points": 0.0,
            "curve_distance_min": np.nan,
            "curve_distance_max": np.nan,
            "curve_distance_coverage": np.nan,
        }
        return x_grid, resampled, stats

    x_observed = numeric[distance_column].to_numpy(dtype=float)
    x_min = float(np.min(x_observed))
    x_max = float(np.max(x_observed))
    within_observed = (x_grid >= x_min) & (x_grid <= x_max)
    for column in value_columns:
        y_observed = numeric[column].to_numpy(dtype=float)
        valid = np.isfinite(y_observed)
        if np.sum(valid) < 2:
            continue
        resampled[column][within_observed] = np.interp(
            x_grid[within_observed],
            x_observed[valid],
            y_observed[valid],
        )

    stats = {
        "curve_n_original_points": float(len(numeric)),
        "curve_distance_min": x_min,
        "curve_distance_max": x_max,
        "curve_distance_coverage": float(x_max - x_min),
    }
    return x_grid, resampled, stats


def _calculate_derivatives(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.full_like(y, np.nan, dtype=float)
    d2 = np.full_like(y, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if valid_indices.size < 2:
        return d1, d2

    valid_x = x[valid_indices]
    valid_y = y[valid_indices]
    edge_order = 2 if valid_indices.size >= 3 else 1
    valid_d1 = np.gradient(valid_y, valid_x, edge_order=edge_order)
    valid_d2 = np.gradient(valid_d1, valid_x, edge_order=edge_order)
    d1[valid_indices] = valid_d1
    d2[valid_indices] = valid_d2
    return d1, d2


def _zone_definitions(n_zones: int) -> list[tuple[str, float, float]]:
    zones: list[tuple[str, float, float]] = [("full", 0.0, 1.0)]
    edges = np.linspace(0.0, 1.0, n_zones + 1)
    default_labels = ("proximal", "middle", "distal") if n_zones == 3 else None
    for index in range(n_zones):
        label = default_labels[index] if default_labels is not None else f"zone{index + 1}"
        zones.append((label, float(edges[index]), float(edges[index + 1])))
    return zones


def _empty_features(prefix: str, names: Iterable[str]) -> dict[str, float]:
    return {f"{prefix}_{name}": np.nan for name in names}


def _ordered_existing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]
