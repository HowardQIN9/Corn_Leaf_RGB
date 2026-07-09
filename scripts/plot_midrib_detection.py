"""Plot midrib valley detection results on green profile curves."""

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


def plot_midrib_detection_file(
    profile_csv: Path,
    line_results: pd.DataFrame,
    leaf_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int = 160,
) -> Path:
    """Create one QC plot showing midrib detection for one leaf image."""
    profile_df = pd.read_csv(profile_csv)
    if profile_df.empty:
        raise ValueError(f"Profile CSV is empty: {profile_csv}")

    source_name = profile_csv.name
    line_df = line_results[line_results["source_profile_file"] == source_name].copy()
    summary_df = leaf_summary[leaf_summary["source_profile_file"] == source_name].copy()
    qc_flag = summary_df["qc_flag"].iloc[0] if not summary_df.empty else "unknown"
    notes = summary_df["notes"].iloc[0] if not summary_df.empty else ""

    sample_ids = sorted(profile_df["sample_id"].dropna().astype(int).unique())
    fig, axes = plt.subplots(1, len(sample_ids), figsize=(3.4 * len(sample_ids), 4.4), sharex=True, sharey=True)
    if len(sample_ids) == 1:
        axes = [axes]

    y_min = float(profile_df["green_mean"].min())
    y_max = float(profile_df["green_mean"].max())
    padding = max(1.0, 0.05 * (y_max - y_min))

    for axis, sample_id in zip(list(axes), sample_ids):
        profile = profile_df[profile_df["sample_id"].astype(int) == sample_id].sort_values("position_fraction")
        axis.plot(profile["position_fraction"], profile["green_mean"], color="#1b8a3a", linewidth=1.4)
        axis.set_title(f"line {sample_id + 1}", fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.set_xlabel("relative position")

        row = line_df[line_df["sample_id"].astype(int) == sample_id]
        if row.empty or row["status"].iloc[0] != "detected":
            axis.text(0.5, 0.92, "no_peak", transform=axis.transAxes, ha="center", va="center", color="#9a3412")
            continue

        result = row.iloc[0]
        peak_x = float(result["peak_position_fraction"])
        split_x = float(result["split_position_fraction"])
        left_x = float(result["peak_left_fraction"])
        right_x = float(result["peak_right_fraction"])
        axis.axvspan(left_x, right_x, color="#fbbf24", alpha=0.22, label="midrib region")
        axis.axvline(peak_x, color="#111827", linestyle="-", linewidth=1.1, label="midrib peak")
        axis.axvline(split_x, color="#dc2626", linestyle="--", linewidth=1.2, label="split")
        axis.text(
            0.5,
            0.92,
            f"split={split_x:.3f}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="#111827",
        )

    list(axes)[0].set_ylabel("green mean")
    title = _title(profile_csv, qc_flag, notes)
    fig.suptitle(title, fontsize=11)
    handles, labels = list(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_csv.stem}_midrib_detection.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_midrib_detection_folder(
    profile_dir: Path,
    line_results_csv: Path,
    leaf_summary_csv: Path,
    output_dir: Path,
    dpi: int = 160,
) -> int:
    """Plot midrib detection QC images for all profile CSVs in a folder."""
    line_results = pd.read_csv(line_results_csv)
    leaf_summary = pd.read_csv(leaf_summary_csv)
    count = 0
    for profile_csv in sorted(profile_dir.glob("*_green_profiles.csv")):
        plot_midrib_detection_file(profile_csv, line_results, leaf_summary, output_dir, dpi)
        count += 1
    return count


def _title(profile_csv: Path, qc_flag: str, notes: str) -> str:
    info = parse_leaf_filename(profile_csv)
    if info is None:
        base = profile_csv.stem
    else:
        base = f"plot {info.plot_number} | {info.geno} | plant {info.plant_number} | {info.leaf}"
    note_text = "" if not isinstance(notes, str) or notes == "" else f" | {notes}"
    return f"{base} | midrib QC: {qc_flag}{note_text}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot midrib detection QC figures.")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Plot manual-adjusted split results instead of automatic detection results.",
    )
    parser.add_argument(
        "--profile_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/profiles"),
    )
    parser.add_argument(
        "--line_results_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/midrib_peak_line_results.csv"),
    )
    parser.add_argument(
        "--leaf_summary_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/midrib_peak_leaf_summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/qc_plots"),
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.manual:
        args.line_results_csv = Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "midrib_peak_line_results_manual_adjusted.csv"
        )
        args.output_dir = Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/manual_qc_plots"
        )
    count = plot_midrib_detection_folder(
        args.profile_dir,
        args.line_results_csv,
        args.leaf_summary_csv,
        args.output_dir,
        args.dpi,
    )
    print(f"Wrote {count} midrib detection QC plots to {args.output_dir}")


if __name__ == "__main__":
    main()
