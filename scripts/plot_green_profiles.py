"""Plot green-channel profile curves for each image.

Each output PNG contains one subplot per sampling line, normally five subplots
for the five leaf positions used in the current workflow.
"""

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

from scripts.extract_leaf2_crops import parse_leaf_filename


def plot_profile_file(
    profile_csv: Path,
    output_dir: Path,
    layout: str = "row",
    dpi: int = 160,
    y_min: float | None = None,
    y_max: float | None = None,
) -> Path:
    """Create one PNG with aligned subplots for all sample lines in one profile CSV."""
    df = pd.read_csv(profile_csv)
    if df.empty:
        raise ValueError(f"Profile CSV is empty: {profile_csv}")

    sample_ids = sorted(df["sample_id"].dropna().astype(int).unique())
    if not sample_ids:
        raise ValueError(f"No sample_id values found in {profile_csv}")

    n_samples = len(sample_ids)
    if layout == "column":
        fig, axes = plt.subplots(n_samples, 1, figsize=(5.5, 2.2 * n_samples), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(1, n_samples, figsize=(3.2 * n_samples, 4.2), sharex=True, sharey=True)
    if n_samples == 1:
        axes = [axes]

    x_values = df["position_fraction"].astype(float)
    y_values = df["green_mean"].astype(float)
    x_limits = (0.0, 1.0)
    y_limits = _axis_limits(y_values, y_min, y_max)

    for axis, sample_id in zip(list(axes), sample_ids):
        profile = df[df["sample_id"].astype(int) == sample_id].sort_values("position_fraction")
        axis.plot(profile["position_fraction"], profile["green_mean"], color="#1b8a3a", linewidth=1.4)
        axis.set_title(f"line {sample_id + 1}", fontsize=10)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.set_xlabel("relative position")
    list(axes)[0].set_ylabel("green mean")

    title = _plot_title(profile_csv)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_csv.stem}_profile_plot.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_profile_folder(
    profile_dir: Path,
    output_dir: Path,
    layout: str = "row",
    dpi: int = 160,
    y_min: float | None = None,
    y_max: float | None = None,
) -> int:
    """Plot every per-image green profile CSV in a folder."""
    profile_paths = sorted(profile_dir.glob("*_green_profiles.csv"))
    count = 0
    for profile_csv in profile_paths:
        plot_profile_file(profile_csv, output_dir, layout, dpi, y_min, y_max)
        count += 1
    return count


def _axis_limits(values: pd.Series, y_min: float | None, y_max: float | None) -> tuple[float, float]:
    lower = float(values.min()) if y_min is None else float(y_min)
    upper = float(values.max()) if y_max is None else float(y_max)
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def _plot_title(profile_csv: Path) -> str:
    info = parse_leaf_filename(profile_csv)
    if info is None:
        return profile_csv.stem
    return (
        f"plot {info.plot_number} | {info.geno} | plant {info.plant_number} | "
        f"{info.leaf} | {info.timestamp}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot green profile curves for each image.")
    parser.add_argument(
        "--profile_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/profiles"),
        help="Folder containing per-image *_green_profiles.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/profile_plots"),
        help="Folder where profile plot PNGs will be saved.",
    )
    parser.add_argument("--layout", choices=["row", "column"], default="row")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = plot_profile_folder(args.profile_dir, args.output_dir, args.layout, args.dpi, args.y_min, args.y_max)
    print(f"Wrote {count} profile plot PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
