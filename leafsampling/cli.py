"""CLI for geometric centerline and normal sampling-line generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from leafsampling.pipeline import CenterlineSamplingConfig, process_mask_folder


def build_parser() -> argparse.ArgumentParser:
    """Build the centerline/sampling CLI parser."""
    parser = argparse.ArgumentParser(description="Generate geometric centerlines and normal sampling lines.")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Folder containing binary mask PNG files.")
    parser.add_argument("--image_dir", type=Path, default=None, help="Optional folder containing matching crop images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for centerline/sampling outputs.")
    parser.add_argument("--min_leaf_width", type=int, default=20)
    parser.add_argument("--tip_trim_ratio", type=float, default=0.02)
    parser.add_argument("--smoothing_window_length", type=int, default=51)
    parser.add_argument("--smoothing_polyorder", type=int, default=2)
    parser.add_argument("--sampling_mode", choices=["fixed_count", "fixed_step"], default="fixed_count")
    parser.add_argument("--n_sampling_lines", type=int, default=5)
    parser.add_argument("--sampling_step", type=int, default=10)
    parser.add_argument("--strip_width_px", type=int, default=5, help="Green profile averaging width in pixels.")
    parser.add_argument("--edge_trim_ratio", type=float, default=0.05)
    parser.add_argument("--boundary_step_size", type=float, default=0.5)
    parser.add_argument("--max_boundary_steps", type=int, default=10000)
    parser.add_argument("--min_sampling_line_length", type=float, default=20.0)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run centerline/sampling generation from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = CenterlineSamplingConfig(
        min_leaf_width=args.min_leaf_width,
        tip_trim_ratio=args.tip_trim_ratio,
        smoothing_window_length=args.smoothing_window_length,
        smoothing_polyorder=args.smoothing_polyorder,
        sampling_mode=args.sampling_mode,
        n_sampling_lines=args.n_sampling_lines,
        sampling_step=args.sampling_step,
        strip_width_px=args.strip_width_px,
        edge_trim_ratio=args.edge_trim_ratio,
        boundary_step_size=args.boundary_step_size,
        max_boundary_steps=args.max_boundary_steps,
        min_sampling_line_length=args.min_sampling_line_length,
    )
    process_mask_folder(args.mask_dir, args.output_dir, config, args.image_dir)


if __name__ == "__main__":
    main()
