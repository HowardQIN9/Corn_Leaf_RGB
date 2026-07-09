"""Combine per-image green profile CSV files into one analysis table.

The combined table keeps every sampled profile point and adds parsed filename
fields such as plot number, genotype, plant number, leaf label, and timestamp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_leaf2_crops import parse_leaf_filename


PARSED_COLUMNS = [
    "prefix",
    "plot_number",
    "geno",
    "plant_number",
    "leaf",
    "timestamp",
]


def combine_green_profiles(profile_dir: Path, output_csv: Path) -> int:
    """Combine all green profile CSV files in profile_dir into output_csv."""
    profile_paths = sorted(profile_dir.glob("*_green_profiles.csv"))
    rows: list[pd.DataFrame] = []
    for profile_path in profile_paths:
        info = parse_leaf_filename(profile_path)
        if info is None:
            continue
        df = pd.read_csv(profile_path)
        if df.empty:
            continue
        df.insert(0, "source_profile_file", profile_path.name)
        for column in reversed(PARSED_COLUMNS):
            df.insert(0, column, getattr(info, column))
        df["plot_number"] = pd.to_numeric(df["plot_number"], errors="coerce").astype("Int64")
        df["plant_number"] = pd.to_numeric(df["plant_number"], errors="coerce").astype("Int64")
        rows.append(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        combined = pd.concat(rows, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=[*PARSED_COLUMNS, "source_profile_file"])
    combined.to_csv(output_csv, index=False)
    return len(combined)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine green profile CSV files into one analysis table.")
    parser.add_argument(
        "--profile_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/profiles"),
        help="Folder containing per-image *_green_profiles.csv files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv"),
        help="Combined output CSV path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_rows = combine_green_profiles(args.profile_dir, args.output_csv)
    print(f"Wrote {n_rows} profile rows to {args.output_csv}")


if __name__ == "__main__":
    main()
