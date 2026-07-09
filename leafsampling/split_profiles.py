"""Split transverse profiles into upper/lower sides starting at midrib boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_profiles_from_midrib_region(profiles: pd.DataFrame, line_results: pd.DataFrame) -> pd.DataFrame:
    """Split each profile line into upper/lower sides using midrib region boundaries.

    Output rows are ordered so relative_distance_from_midrib starts at 0 near
    the midrib region boundary and increases toward the leaf edge.
    """
    required_profiles = {"source_profile_file", "sample_id", "position_fraction"}
    required_lines = {"source_profile_file", "sample_id", "peak_left_fraction", "peak_right_fraction"}
    missing_profiles = required_profiles - set(profiles.columns)
    missing_lines = required_lines - set(line_results.columns)
    if missing_profiles:
        raise ValueError(f"Missing profile columns: {sorted(missing_profiles)}")
    if missing_lines:
        raise ValueError(f"Missing line result columns: {sorted(missing_lines)}")

    rows: list[pd.DataFrame] = []
    line_lookup = line_results.set_index(["source_profile_file", "sample_id"], drop=False)
    for (source_profile_file, sample_id), profile in profiles.groupby(["source_profile_file", "sample_id"], sort=True):
        if (source_profile_file, sample_id) not in line_lookup.index:
            continue
        line = line_lookup.loc[(source_profile_file, sample_id)]
        if isinstance(line, pd.DataFrame):
            line = line.iloc[0]
        left = pd.to_numeric(pd.Series([line["peak_left_fraction"]]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([line["peak_right_fraction"]]), errors="coerce").iloc[0]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        left = float(np.clip(left, 0.0, 1.0))
        right = float(np.clip(right, 0.0, 1.0))
        if right < left:
            left, right = right, left

        upper = _make_side_profile(profile, "upper", boundary_fraction=left, edge_fraction=0.0)
        lower = _make_side_profile(profile, "lower", boundary_fraction=right, edge_fraction=1.0)
        for side_df in (upper, lower):
            if side_df.empty:
                continue
            side_df["midrib_boundary_fraction"] = left if side_df["midrib_side"].iloc[0] == "upper" else right
            side_df["midrib_region_left_fraction"] = left
            side_df["midrib_region_right_fraction"] = right
            side_df["sample_id"] = int(sample_id)
            rows.append(side_df)

    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    return output.sort_values(["source_profile_file", "sample_id", "midrib_side", "distance_index"]).reset_index(drop=True)


def _make_side_profile(
    profile: pd.DataFrame,
    midrib_side: str,
    boundary_fraction: float,
    edge_fraction: float,
) -> pd.DataFrame:
    position = profile["position_fraction"].astype(float)
    if midrib_side == "upper":
        side = profile[position <= boundary_fraction].copy()
        side["distance_from_midrib"] = boundary_fraction - side["position_fraction"].astype(float)
    elif midrib_side == "lower":
        side = profile[position >= boundary_fraction].copy()
        side["distance_from_midrib"] = side["position_fraction"].astype(float) - boundary_fraction
    else:
        raise ValueError(f"Unsupported midrib side: {midrib_side}")

    if side.empty:
        return side
    max_distance = abs(float(edge_fraction) - float(boundary_fraction))
    side["relative_distance_from_midrib"] = (
        side["distance_from_midrib"] / max_distance if max_distance > 0 else 0.0
    )
    side = side.sort_values("distance_from_midrib", ascending=True).reset_index(drop=True)
    side["distance_index"] = np.arange(len(side))
    side["midrib_side"] = midrib_side
    return side
