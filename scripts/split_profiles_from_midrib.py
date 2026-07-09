"""Split profiles into upper/lower sides from midrib region boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.split_profiles import split_profiles_from_midrib_region
from scripts.extract_leaf2_crops import parse_leaf_filename


def split_and_save(profiles_csv: Path, line_results_csv: Path, output_csv: Path) -> int:
    """Split combined profiles into midrib-outward upper/lower profiles."""
    profiles = pd.read_csv(profiles_csv)
    line_results = pd.read_csv(line_results_csv)
    split_profiles = split_profiles_from_midrib_region(profiles, line_results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_profiles.to_csv(output_csv, index=False)
    return len(split_profiles)


def plot_split_profiles(split_profiles_csv: Path, output_dir: Path, dpi: int = 160) -> int:
    """Plot one 5x2 upper/lower QC figure per leaf."""
    split_profiles = pd.read_csv(split_profiles_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_profile_file, leaf_df in split_profiles.groupby("source_profile_file", sort=True):
        _plot_one_leaf(source_profile_file, leaf_df, output_dir, dpi)
        count += 1
    return count


def _plot_one_leaf(source_profile_file: str, leaf_df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    sample_ids = sorted(leaf_df["sample_id"].dropna().astype(int).unique())
    fig, axes = plt.subplots(len(sample_ids), 2, figsize=(8.8, 2.1 * len(sample_ids)), sharex=True, sharey=True)
    if len(sample_ids) == 1:
        axes = [axes]

    y_min = float(leaf_df["green_mean"].min())
    y_max = float(leaf_df["green_mean"].max())
    padding = max(1.0, 0.05 * (y_max - y_min))

    for row_index, sample_id in enumerate(sample_ids):
        for col_index, side in enumerate(["upper", "lower"]):
            axis = axes[row_index][col_index]
            profile = leaf_df[
                (leaf_df["sample_id"].astype(int) == sample_id) & (leaf_df["midrib_side"] == side)
            ].sort_values("distance_index")
            axis.plot(profile["relative_distance_from_midrib"], profile["green_mean"], color="#1b8a3a", linewidth=1.2)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(y_min - padding, y_max + padding)
            axis.grid(True, color="#dddddd", linewidth=0.6)
            axis.set_title(f"line {sample_id + 1} | {side}", fontsize=9)
            if row_index == len(sample_ids) - 1:
                axis.set_xlabel("distance from midrib boundary")
            if col_index == 0:
                axis.set_ylabel("green mean")

    fig.suptitle(_title(source_profile_file), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / f"{Path(source_profile_file).stem}_split_upper_lower.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _title(source_profile_file: str) -> str:
    info = parse_leaf_filename(Path(source_profile_file))
    if info is None:
        return source_profile_file
    return f"plot {info.plot_number} | {info.geno} | plant {info.plant_number} | {info.leaf}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split profiles into upper/lower sides from midrib boundaries.")
    parser.add_argument(
        "--profiles_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv"),
    )
    parser.add_argument(
        "--line_results_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "midrib_peak_line_results_manual_adjusted.csv"
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "all_leaf2_green_profiles_split_from_midrib.csv"
        ),
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/split_profile_plots"),
    )
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_rows = split_and_save(args.profiles_csv, args.line_results_csv, args.output_csv)
    print(f"Wrote {n_rows} split profile rows to {args.output_csv}")
    if not args.skip_plots:
        n_plots = plot_split_profiles(args.output_csv, args.plot_dir, args.dpi)
        print(f"Wrote {n_plots} split profile plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
