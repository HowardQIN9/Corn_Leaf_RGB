"""Extract curve-level and image-level features from separated profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.features import aggregate_image_features, extract_curve_features


DEFAULT_INPUT = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
    "all_leaf2_green_profiles_split_meso_peak.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/feature_analysis")


def extract_and_save(
    input_csv: Path,
    curve_output_csv: Path,
    image_output_csv: Path | None,
    *,
    value_columns: tuple[str, ...],
    derivative_column: str | None,
    n_zones: int,
    resample_points: int,
    min_points: int,
    peak_prominence_fraction: float,
    image_aggregations: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read profiles, extract features, and write requested output tables."""
    profiles = pd.read_csv(input_csv)
    curve_features = extract_curve_features(
        profiles,
        value_columns=value_columns,
        derivative_column=derivative_column,
        n_zones=n_zones,
        resample_points=resample_points,
        min_points=min_points,
        peak_prominence_fraction=peak_prominence_fraction,
    )
    curve_output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_features.to_csv(curve_output_csv, index=False)

    image_features: pd.DataFrame | None = None
    if image_output_csv is not None:
        image_features = aggregate_image_features(curve_features, aggregations=image_aggregations)
        image_output_csv.parent.mkdir(parents=True, exist_ok=True)
        image_features.to_csv(image_output_csv, index=False)
    return curve_features, image_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract signal, derivative, curvature, and peak features from split leaf profiles."
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--curve_output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "leaf2_green_curve_features.csv",
        help="One output row per sample_id and midrib_side.",
    )
    parser.add_argument(
        "--image_output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "leaf2_green_image_features.csv",
        help="One output row per source image, aggregated across its curves.",
    )
    parser.add_argument(
        "--value_columns",
        nargs="+",
        default=["green_mean", "green_mean_meso", "green_mean_peak"],
    )
    parser.add_argument(
        "--derivative_column",
        default="green_mean_meso",
        help="Column used for d1, d2, and curvature features; use 'none' to disable.",
    )
    parser.add_argument("--n_zones", type=int, default=3)
    parser.add_argument("--resample_points", type=int, default=256)
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument(
        "--peak_prominence_fraction",
        type=float,
        default=0.10,
        help="Peak prominence threshold as a fraction of the zone's robust 5-95%% range.",
    )
    parser.add_argument(
        "--image_aggregations",
        nargs="+",
        choices=["mean", "std", "min", "max", "median"],
        default=["mean", "std", "min", "max"],
    )
    parser.add_argument(
        "--skip_image_summary",
        action="store_true",
        help="Write only the curve-level table.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    derivative_column = None if args.derivative_column.lower() == "none" else args.derivative_column
    image_output_csv = None if args.skip_image_summary else args.image_output_csv
    curve_features, image_features = extract_and_save(
        args.input_csv,
        args.curve_output_csv,
        image_output_csv,
        value_columns=tuple(args.value_columns),
        derivative_column=derivative_column,
        n_zones=args.n_zones,
        resample_points=args.resample_points,
        min_points=args.min_points,
        peak_prominence_fraction=args.peak_prominence_fraction,
        image_aggregations=tuple(args.image_aggregations),
    )
    print(
        f"Wrote {len(curve_features)} curve rows and {curve_features.shape[1]} columns "
        f"to {args.curve_output_csv}"
    )
    if image_features is not None and image_output_csv is not None:
        print(
            f"Wrote {len(image_features)} image rows and {image_features.shape[1]} columns "
            f"to {image_output_csv}"
        )


if __name__ == "__main__":
    main()
