"""Copy segmented Leaf2 crop outputs into a separate folder.

Filename example:
LeafDoc_109_tall_1_Leaf3_1780576981689_leaf_crop.tif

Parsed fields:
- LeafDoc: dataset/source prefix
- 109: plot number
- tall: genotype/group
- 1: plant number inside plot, usually 1-5
- Leaf3: leaf number, usually Leaf1-Leaf3
- 1780576981689: timestamp/id
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shutil


FILENAME_RE = re.compile(
    r"^(?P<prefix>[^_]+)_"
    r"(?P<plot_number>\d+)_"
    r"(?P<geno>[^_]+)_"
    r"(?P<plant_number>\d+)_"
    r"(?P<leaf>Leaf\d+)_"
    r"(?P<timestamp>\d+)"
    r"(?P<output_suffix>_leaf_crop(?:_mask)?|_leaf_mask|_overlay|_segmented_preview|_green_profiles|_sampling_lines|_centerline)?$"
)


@dataclass(frozen=True)
class LeafFileInfo:
    """Information parsed from one segmented leaf filename."""

    prefix: str
    plot_number: str
    geno: str
    plant_number: str
    leaf: str
    timestamp: str
    output_suffix: str


def parse_leaf_filename(path: Path) -> LeafFileInfo | None:
    """Parse a segmented leaf output filename.

    Returns None when the filename does not match the expected convention.
    """
    match = FILENAME_RE.match(path.stem)
    if match is None:
        return None
    groups = match.groupdict()
    return LeafFileInfo(
        prefix=groups["prefix"],
        plot_number=groups["plot_number"],
        geno=groups["geno"],
        plant_number=groups["plant_number"],
        leaf=groups["leaf"],
        timestamp=groups["timestamp"],
        output_suffix=groups["output_suffix"] or "",
    )


def copy_leaf_files(input_dir: Path, output_dir: Path, leaf: str = "Leaf2") -> int:
    """Copy all files for one leaf label from input_dir to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        info = parse_leaf_filename(path)
        if info is None or info.leaf != leaf:
            continue
        destination = output_dir / path.name
        shutil.copy2(path, destination)
        copied += 1
    return copied


def write_filtered_metadata(metadata_csv: Path, output_csv: Path, leaf: str = "Leaf2") -> int:
    """Write a metadata CSV containing only rows for one leaf label."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with metadata_csv.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        with output_csv.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                filename = row.get("filename", "")
                info = parse_leaf_filename(Path(filename))
                if info is None or info.leaf != leaf:
                    continue
                writer.writerow(row)
                kept += 1
    return kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy segmented Leaf2 outputs into a separate folder.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_run/crops"),
        help="Folder containing segmented crop outputs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_crops"),
        help="Folder where Leaf2 crop outputs will be copied.",
    )
    parser.add_argument("--leaf", default="Leaf2", help="Leaf label to copy, for example Leaf1, Leaf2, or Leaf3.")
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("outputs/RGB_tall_v9_run/metadata/segmentation_metadata.csv"),
        help="Optional metadata CSV to filter.",
    )
    parser.add_argument(
        "--metadata_output",
        type=Path,
        default=Path("outputs/RGB_tall_v9_leaf2_crops/leaf2_metadata.csv"),
        help="Filtered metadata output path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    copied = copy_leaf_files(args.input_dir, args.output_dir, args.leaf)
    print(f"Copied {copied} files for {args.leaf} to {args.output_dir}")

    if args.metadata_csv.exists():
        rows = write_filtered_metadata(args.metadata_csv, args.metadata_output, args.leaf)
        print(f"Wrote {rows} metadata rows to {args.metadata_output}")
    else:
        print(f"Metadata CSV not found, skipped: {args.metadata_csv}")


if __name__ == "__main__":
    main()
