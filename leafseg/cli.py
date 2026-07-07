"""Command-line interface for corn leaf segmentation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from leafseg.config import SegmentationConfig, load_config, merge_config
from leafseg.pipeline import process_folder


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Segment corn leaves from TIFF/JPEG/PNG images.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for segmentation outputs.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file.")
    parser.add_argument("--method", choices=["hsv", "exg", "hsv_exg_fallback", "hsv+exg"], default=None)
    parser.add_argument("--h_min", type=int, default=None)
    parser.add_argument("--h_max", type=int, default=None)
    parser.add_argument("--s_min", type=int, default=None)
    parser.add_argument("--v_min", type=int, default=None)
    parser.add_argument("--exg_threshold", type=float, default=None)
    parser.add_argument("--morph_kernel_size", type=int, default=None)
    parser.add_argument("--min_leaf_area_pixels", type=int, default=None)
    parser.add_argument("--min_mask_area_ratio", type=float, default=None)
    parser.add_argument("--max_mask_area_ratio", type=float, default=None)
    parser.add_argument("--max_bbox_aspect_ratio", type=float, default=None)
    parser.add_argument("--bbox_padding", type=int, default=None)
    parser.add_argument("--save_segmented_preview", type=_parse_bool, default=None)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the segmentation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config) if args.config else SegmentationConfig()
    overrides: dict[str, Any] = {
        key: getattr(args, key)
        for key in (
            "method",
            "h_min",
            "h_max",
            "s_min",
            "v_min",
            "exg_threshold",
            "morph_kernel_size",
            "min_leaf_area_pixels",
            "min_mask_area_ratio",
            "max_mask_area_ratio",
            "max_bbox_aspect_ratio",
            "bbox_padding",
            "save_segmented_preview",
        )
    }
    config = merge_config(config, overrides)
    process_folder(args.input_dir, args.output_dir, config)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


if __name__ == "__main__":
    main()
