"""Interactively adjust midrib split points by clicking profile plots.

Example:
python scripts/interactive_adjust_midrib.py --image_name LeafDoc_212_tall_1_Leaf2

Click inside a subplot to set the manual split for that sampling line.
Close the figure window to save edits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.manual_adjust_midrib import export_manual_template


def find_profile_file(profile_dir: Path, image_name: str) -> Path:
    """Find one profile CSV whose filename contains image_name."""
    matches = sorted(profile_dir.glob(f"*{image_name}*_green_profiles.csv"))
    if not matches:
        raise FileNotFoundError(f"No profile CSV matched: {image_name}")
    if len(matches) > 1:
        names = "\n".join(path.name for path in matches[:20])
        raise ValueError(f"Multiple profile CSVs matched '{image_name}'. Be more specific:\n{names}")
    return matches[0]


def ensure_manual_csv(line_results_csv: Path, manual_csv: Path) -> None:
    """Create the manual adjustment CSV if it does not exist."""
    if not manual_csv.exists():
        export_manual_template(line_results_csv, manual_csv)


def load_line_results_for_profile(line_results_csv: Path, source_profile_file: str) -> pd.DataFrame:
    """Load automatic line results for one profile file."""
    line_results = pd.read_csv(line_results_csv)
    return line_results[line_results["source_profile_file"] == source_profile_file].copy()


def update_manual_split(
    manual_csv: Path,
    source_profile_file: str,
    sample_id: int,
    split_position_fraction: float,
    note: str = "interactive click",
) -> None:
    """Update one row in the manual adjustment CSV."""
    if not 0.0 <= split_position_fraction <= 1.0:
        raise ValueError(f"split_position_fraction must be in [0, 1], got {split_position_fraction}")
    manual = pd.read_csv(manual_csv)
    for column in ("manual_split_position_fraction", "manual_notes"):
        if column in manual.columns:
            manual[column] = manual[column].astype("object")
    mask = (manual["source_profile_file"] == source_profile_file) & (manual["sample_id"].astype(int) == sample_id)
    if not bool(mask.any()):
        raise ValueError(f"No manual adjustment row for {source_profile_file}, sample_id={sample_id}")
    manual.loc[mask, "manual_split_position_fraction"] = float(split_position_fraction)
    manual.loc[mask, "use_manual"] = True
    manual.loc[mask, "manual_notes"] = note
    manual.to_csv(manual_csv, index=False)


def interactive_adjust_profile(
    profile_csv: Path,
    line_results_csv: Path,
    manual_csv: Path,
) -> None:
    """Open an interactive plot and save click-based manual split edits."""
    profile_df = pd.read_csv(profile_csv)
    line_df = load_line_results_for_profile(line_results_csv, profile_csv.name)
    sample_ids = sorted(profile_df["sample_id"].dropna().astype(int).unique())
    if not sample_ids:
        raise ValueError(f"No sample_id values in {profile_csv}")

    fig, axes = plt.subplots(1, len(sample_ids), figsize=(3.4 * len(sample_ids), 4.4), sharex=True, sharey=True)
    if len(sample_ids) == 1:
        axes = [axes]

    y_min = float(profile_df["green_mean"].min())
    y_max = float(profile_df["green_mean"].max())
    padding = max(1.0, 0.05 * (y_max - y_min))
    axis_to_sample: dict[object, int] = {}
    click_lines: dict[int, object] = {}

    for axis, sample_id in zip(list(axes), sample_ids):
        axis_to_sample[axis] = sample_id
        profile = profile_df[profile_df["sample_id"].astype(int) == sample_id].sort_values("position_fraction")
        axis.plot(profile["position_fraction"], profile["green_mean"], color="#1b8a3a", linewidth=1.4)
        axis.set_title(f"line {sample_id + 1}", fontsize=10)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.set_xlabel("relative position")

        auto = line_df[line_df["sample_id"].astype(int) == sample_id]
        if not auto.empty and auto["status"].iloc[0] == "detected":
            auto_split = float(auto["split_position_fraction"].iloc[0])
            axis.axvline(auto_split, color="#111827", linestyle="-", linewidth=1.0, label="auto")

        manual = pd.read_csv(manual_csv)
        row = manual[(manual["source_profile_file"] == profile_csv.name) & (manual["sample_id"].astype(int) == sample_id)]
        if not row.empty and _as_bool(row["use_manual"].iloc[0]):
            value = pd.to_numeric(row["manual_split_position_fraction"], errors="coerce").iloc[0]
            if pd.notna(value):
                click_lines[sample_id] = axis.axvline(float(value), color="#dc2626", linestyle="--", linewidth=1.4)
                axis.text(0.5, 0.92, f"manual={float(value):.3f}", transform=axis.transAxes, ha="center")

    list(axes)[0].set_ylabel("green mean")
    fig.suptitle(f"{profile_csv.name}\nClick a subplot to set manual split. Close window to finish.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    def on_click(event) -> None:
        if event.inaxes not in axis_to_sample or event.xdata is None:
            return
        split = max(0.0, min(1.0, float(event.xdata)))
        sample_id = axis_to_sample[event.inaxes]
        update_manual_split(manual_csv, profile_csv.name, sample_id, split)
        if sample_id in click_lines:
            click_lines[sample_id].remove()
        click_lines[sample_id] = event.inaxes.axvline(split, color="#dc2626", linestyle="--", linewidth=1.4)
        event.inaxes.set_title(f"line {sample_id + 1} manual={split:.3f}", fontsize=10)
        fig.canvas.draw_idle()
        print(f"Updated {profile_csv.name} sample_id={sample_id} manual split -> {split:.4f}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Click profile plots to manually adjust midrib split points.")
    parser.add_argument("--image_name", required=True, help="Full or partial image/profile name to open.")
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
        "--manual_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/manual_midrib_split_adjustments.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_manual_csv(args.line_results_csv, args.manual_csv)
    profile_csv = find_profile_file(args.profile_dir, args.image_name)
    print(f"Opening {profile_csv.name}")
    interactive_adjust_profile(profile_csv, args.line_results_csv, args.manual_csv)
    print(f"Saved manual edits in {args.manual_csv}")
    print("Run this next to apply all manual edits:")
    print("conda run -n corn python scripts\\manual_adjust_midrib.py apply")


if __name__ == "__main__":
    main()
