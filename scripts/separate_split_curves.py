"""Separate split upper/lower profiles into meso baseline and peak residual curves."""

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

from leafsampling.curve_separation import separate_profile_components
from scripts.extract_leaf2_crops import parse_leaf_filename


def separate_and_save(
    input_csv: Path,
    output_csv: Path,
    *,
    value_column: str = "green_mean",
    valley_distance: int = 15,
    smooth_window: int = 51,
) -> int:
    """Read split profiles, add meso/peak columns, and save a long CSV."""
    split_profiles = pd.read_csv(input_csv)
    separated = separate_profile_components(
        split_profiles,
        value_column=value_column,
        valley_distance=valley_distance,
        smooth_window=smooth_window,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    separated.to_csv(output_csv, index=False)
    return len(separated)


def plot_separated_curves(
    separated_csv: Path,
    output_dir: Path,
    *,
    value_column: str = "green_mean",
    dpi: int = 160,
) -> int:
    """Plot one 5x2 QC figure per leaf for the residual peak curves."""
    separated = pd.read_csv(separated_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_profile_file, leaf_df in separated.groupby("source_profile_file", sort=True):
        _plot_one_leaf(source_profile_file, leaf_df, output_dir, value_column, dpi)
        count += 1
    return count


def plot_meso_curves(
    separated_csv: Path,
    output_dir: Path,
    *,
    value_column: str = "green_mean",
    dpi: int = 160,
) -> int:
    """Plot one 5x2 QC figure per leaf with raw curves and meso baselines."""
    separated = pd.read_csv(separated_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source_profile_file, leaf_df in separated.groupby("source_profile_file", sort=True):
        _plot_one_leaf_meso(source_profile_file, leaf_df, output_dir, value_column, dpi)
        count += 1
    return count


def _plot_one_leaf(
    source_profile_file: str,
    leaf_df: pd.DataFrame,
    output_dir: Path,
    value_column: str,
    dpi: int,
) -> Path:
    peak_col = f"{value_column}_peak"
    sample_ids = sorted(leaf_df["sample_id"].dropna().astype(int).unique())
    fig, axes = plt.subplots(len(sample_ids), 2, figsize=(9.2, 2.0 * len(sample_ids)), sharex=True, sharey=True)
    if len(sample_ids) == 1:
        axes = [axes]

    y_min = float(leaf_df[peak_col].min())
    y_max = float(leaf_df[peak_col].max())
    padding = max(1.0, 0.05 * (y_max - y_min))

    for row_index, sample_id in enumerate(sample_ids):
        for col_index, side in enumerate(["upper", "lower"]):
            axis = axes[row_index][col_index]
            profile = leaf_df[
                (leaf_df["sample_id"].astype(int) == sample_id) & (leaf_df["midrib_side"] == side)
            ].sort_values("distance_index")
            axis.plot(
                profile["relative_distance_from_midrib"],
                profile[peak_col],
                color="#7a3b9d",
                linewidth=1.1,
            )
            axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.7)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(y_min - padding, y_max + padding)
            axis.grid(True, color="#dddddd", linewidth=0.6)
            axis.set_title(f"line {sample_id + 1} | {side}", fontsize=9)
            if row_index == len(sample_ids) - 1:
                axis.set_xlabel("distance from midrib boundary")
            if col_index == 0:
                axis.set_ylabel(peak_col)

    fig.suptitle(_title(source_profile_file), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / f"{Path(source_profile_file).stem}_green_peak.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _plot_one_leaf_meso(
    source_profile_file: str,
    leaf_df: pd.DataFrame,
    output_dir: Path,
    value_column: str,
    dpi: int,
) -> Path:
    meso_col = f"{value_column}_meso"
    sample_ids = sorted(leaf_df["sample_id"].dropna().astype(int).unique())
    fig, axes = plt.subplots(len(sample_ids), 2, figsize=(9.2, 2.0 * len(sample_ids)), sharex=True, sharey=True)
    if len(sample_ids) == 1:
        axes = [axes]

    y_min = float(leaf_df[[value_column, meso_col]].min().min())
    y_max = float(leaf_df[[value_column, meso_col]].max().max())
    padding = max(1.0, 0.05 * (y_max - y_min))

    for row_index, sample_id in enumerate(sample_ids):
        for col_index, side in enumerate(["upper", "lower"]):
            axis = axes[row_index][col_index]
            profile = leaf_df[
                (leaf_df["sample_id"].astype(int) == sample_id) & (leaf_df["midrib_side"] == side)
            ].sort_values("distance_index")
            axis.plot(
                profile["relative_distance_from_midrib"],
                profile[value_column],
                color="#1b8a3a",
                linewidth=0.9,
                alpha=0.65,
                label=value_column,
            )
            axis.plot(
                profile["relative_distance_from_midrib"],
                profile[meso_col],
                color="#c45a12",
                linewidth=1.2,
                label=meso_col,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(y_min - padding, y_max + padding)
            axis.grid(True, color="#dddddd", linewidth=0.6)
            axis.set_title(f"line {sample_id + 1} | {side}", fontsize=9)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=7, loc="best")
            if row_index == len(sample_ids) - 1:
                axis.set_xlabel("distance from midrib boundary")
            if col_index == 0:
                axis.set_ylabel(value_column)

    fig.suptitle(_title(source_profile_file), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = output_dir / f"{Path(source_profile_file).stem}_green_meso.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _title(source_profile_file: str) -> str:
    info = parse_leaf_filename(Path(source_profile_file))
    if info is None:
        return source_profile_file
    return f"plot {info.plot_number} | {info.geno} | plant {info.plant_number} | {info.leaf}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Separate split profiles into meso baseline and peak residual curves.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "all_leaf2_green_profiles_split_from_midrib.csv"
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "all_leaf2_green_profiles_split_meso_peak.csv"
        ),
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/split_curve_separation_plots"),
    )
    parser.add_argument(
        "--meso_plot_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/split_curve_meso_plots"),
    )
    parser.add_argument("--value_column", default="green_mean")
    parser.add_argument("--valley_distance", type=int, default=15)
    parser.add_argument("--smooth_window", type=int, default=51)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    n_rows = separate_and_save(
        args.input_csv,
        args.output_csv,
        value_column=args.value_column,
        valley_distance=args.valley_distance,
        smooth_window=args.smooth_window,
    )
    print(f"Wrote {n_rows} separated profile rows to {args.output_csv}")
    if not args.skip_plots:
        n_plots = plot_separated_curves(args.output_csv, args.plot_dir, value_column=args.value_column, dpi=args.dpi)
        print(f"Wrote {n_plots} curve-separation plots to {args.plot_dir}")
        n_meso_plots = plot_meso_curves(
            args.output_csv,
            args.meso_plot_dir,
            value_column=args.value_column,
            dpi=args.dpi,
        )
        print(f"Wrote {n_meso_plots} meso baseline plots to {args.meso_plot_dir}")


if __name__ == "__main__":
    main()
