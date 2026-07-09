"""Detect broad midrib-related peaks in green profile curves."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.midrib import (
    MidribDetectionConfig,
    annotate_profile_sides,
    detect_midrib_peaks,
    summarize_midrib_detection,
)


def run_midrib_detection(input_csv: Path, output_dir: Path, config: MidribDetectionConfig) -> dict[str, Path]:
    """Run midrib peak detection and write line, leaf, and annotated profile CSVs."""
    profiles = pd.read_csv(input_csv)
    line_results = detect_midrib_peaks(profiles, config)
    leaf_summary = summarize_midrib_detection(line_results, config)
    annotated_profiles = annotate_profile_sides(profiles, line_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "line_results": output_dir / "midrib_peak_line_results.csv",
        "leaf_summary": output_dir / "midrib_peak_leaf_summary.csv",
        "annotated_profiles": output_dir / "all_leaf2_green_profiles_with_midrib_sides.csv",
    }
    line_results.to_csv(paths["line_results"], index=False)
    leaf_summary.to_csv(paths["leaf_summary"], index=False)
    annotated_profiles.to_csv(paths["annotated_profiles"], index=False)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect constrained broad midrib-related peaks.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection"),
    )
    parser.add_argument("--peak_polarity", choices=["dark", "bright"], default="bright")
    parser.add_argument("--middle_min_fraction", type=float, default=0.35)
    parser.add_argument("--middle_max_fraction", type=float, default=0.70)
    parser.add_argument("--expected_peak_fraction", type=float, default=0.50)
    parser.add_argument("--center_weight_sigma", type=float, default=0.12)
    parser.add_argument("--smoothing_window_length", type=int, default=15)
    parser.add_argument("--smoothing_polyorder", type=int, default=2)
    parser.add_argument("--min_prominence", type=float, default=6.0)
    parser.add_argument("--min_width_fraction", type=float, default=0.01)
    parser.add_argument("--max_peak_position_range", type=float, default=0.20)
    parser.add_argument("--min_detected_lines", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MidribDetectionConfig(
        middle_min_fraction=args.middle_min_fraction,
        peak_polarity=args.peak_polarity,
        middle_max_fraction=args.middle_max_fraction,
        expected_peak_fraction=args.expected_peak_fraction,
        center_weight_sigma=args.center_weight_sigma,
        smoothing_window_length=args.smoothing_window_length,
        smoothing_polyorder=args.smoothing_polyorder,
        min_prominence=args.min_prominence,
        min_width_fraction=args.min_width_fraction,
        max_peak_position_range=args.max_peak_position_range,
        min_detected_lines=args.min_detected_lines,
    )
    paths = run_midrib_detection(args.input_csv, args.output_dir, config)
    print(f"Wrote line results to {paths['line_results']}")
    print(f"Wrote leaf summary to {paths['leaf_summary']}")
    print(f"Wrote annotated profiles to {paths['annotated_profiles']}")


if __name__ == "__main__":
    main()
