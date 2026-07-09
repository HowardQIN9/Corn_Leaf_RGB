"""Create and apply manual midrib split-point adjustments.

Workflow:
1. Export a manual adjustment template CSV.
2. Edit manual_split_position_fraction and use_manual in that CSV.
3. Apply the edited CSV to regenerate side-of-midrib annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.midrib import annotate_profile_sides


def export_manual_template(line_results_csv: Path, output_csv: Path) -> int:
    """Export a line-level CSV that can be manually edited."""
    line_results = pd.read_csv(line_results_csv)
    template = line_results.copy()
    if "manual_split_position_fraction" not in template:
        template["manual_split_position_fraction"] = ""
    if "use_manual" not in template:
        template["use_manual"] = False
    if "manual_notes" not in template:
        template["manual_notes"] = ""

    keep_columns = [
        "prefix",
        "plot_number",
        "geno",
        "plant_number",
        "leaf",
        "timestamp",
        "source_profile_file",
        "filename",
        "sample_id",
        "status",
        "peak_position_fraction",
        "split_position_fraction",
        "peak_left_fraction",
        "peak_right_fraction",
        "peak_prominence",
        "peak_width_fraction",
        "manual_split_position_fraction",
        "use_manual",
        "manual_notes",
        "notes",
    ]
    available = [column for column in keep_columns if column in template.columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    template[available].to_csv(output_csv, index=False)
    return len(template)


def apply_manual_adjustments(
    profiles_csv: Path,
    line_results_csv: Path,
    manual_csv: Path,
    output_profiles_csv: Path,
    output_line_results_csv: Path,
) -> int:
    """Apply manual split points and write adjusted profile annotations."""
    profiles = pd.read_csv(profiles_csv)
    line_results = pd.read_csv(line_results_csv)
    manual = pd.read_csv(manual_csv)

    adjusted_line_results = _merge_manual_adjustments(line_results, manual)
    annotated = annotate_profile_sides(profiles, adjusted_line_results)
    annotated = annotated.merge(
        adjusted_line_results[
            [
                "source_profile_file",
                "sample_id",
                "split_source",
                "manual_notes",
            ]
        ],
        on=["source_profile_file", "sample_id"],
        how="left",
    )

    output_profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    output_line_results_csv.parent.mkdir(parents=True, exist_ok=True)
    adjusted_line_results.to_csv(output_line_results_csv, index=False)
    annotated.to_csv(output_profiles_csv, index=False)
    return int((adjusted_line_results["split_source"] == "manual").sum())


def _merge_manual_adjustments(line_results: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["source_profile_file", "sample_id"]
    manual_cols = [*key_cols, "manual_split_position_fraction", "use_manual", "manual_notes"]
    missing = set(manual_cols) - set(manual.columns)
    if missing:
        raise ValueError(f"Manual adjustment CSV is missing columns: {sorted(missing)}")

    merged = line_results.merge(manual[manual_cols], on=key_cols, how="left", suffixes=("", "_manual_file"))
    merged["split_source"] = "auto"
    merged["manual_notes"] = merged["manual_notes"].fillna("")

    use_manual = merged["use_manual"].map(_as_bool).fillna(False)
    manual_split = pd.to_numeric(merged["manual_split_position_fraction"], errors="coerce")
    valid_manual = use_manual & manual_split.between(0.0, 1.0)

    merged.loc[valid_manual, "split_position_fraction"] = manual_split[valid_manual]
    merged.loc[valid_manual, "peak_position_fraction"] = manual_split[valid_manual]
    merged.loc[valid_manual, "status"] = "detected"
    merged.loc[valid_manual, "split_source"] = "manual"

    # Recenter the visible midrib region around the manually chosen split.
    left = pd.to_numeric(merged.get("peak_left_fraction"), errors="coerce")
    right = pd.to_numeric(merged.get("peak_right_fraction"), errors="coerce")
    region_width = (right - left).where((left.between(0.0, 1.0)) & (right.between(0.0, 1.0)), 0.02)
    half_width = (region_width / 2.0).clip(lower=0.01)
    merged.loc[valid_manual, "peak_left_fraction"] = np.maximum(manual_split[valid_manual] - half_width[valid_manual], 0.0)
    merged.loc[valid_manual, "peak_right_fraction"] = np.minimum(manual_split[valid_manual] + half_width[valid_manual], 1.0)

    invalid_requested = use_manual & ~valid_manual
    if invalid_requested.any():
        bad = merged.loc[invalid_requested, key_cols].to_dict("records")
        raise ValueError(f"Some rows requested manual split but have invalid values: {bad[:5]}")

    drop_cols = ["manual_split_position_fraction", "use_manual"]
    return merged.drop(columns=[column for column in drop_cols if column in merged.columns])


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually adjust midrib split points.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-template")
    export_parser.add_argument(
        "--line_results_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/midrib_peak_line_results.csv"),
    )
    export_parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/manual_midrib_split_adjustments.csv"),
    )

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument(
        "--profiles_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv"),
    )
    apply_parser.add_argument(
        "--line_results_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/midrib_peak_line_results.csv"),
    )
    apply_parser.add_argument(
        "--manual_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/manual_midrib_split_adjustments.csv"),
    )
    apply_parser.add_argument(
        "--output_profiles_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "all_leaf2_green_profiles_with_manual_midrib_sides.csv"
        ),
    )
    apply_parser.add_argument(
        "--output_line_results_csv",
        type=Path,
        default=Path(
            "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
            "midrib_peak_line_results_manual_adjusted.csv"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "export-template":
        n_rows = export_manual_template(args.line_results_csv, args.output_csv)
        print(f"Wrote {n_rows} manual adjustment rows to {args.output_csv}")
    elif args.command == "apply":
        n_manual = apply_manual_adjustments(
            args.profiles_csv,
            args.line_results_csv,
            args.manual_csv,
            args.output_profiles_csv,
            args.output_line_results_csv,
        )
        print(f"Applied {n_manual} manual split adjustments")
        print(f"Wrote adjusted profiles to {args.output_profiles_csv}")
        print(f"Wrote adjusted line results to {args.output_line_results_csv}")


if __name__ == "__main__":
    main()
