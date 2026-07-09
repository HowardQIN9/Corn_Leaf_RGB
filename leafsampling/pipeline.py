"""Batch pipeline for geometric centerlines and normal sampling lines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd

from leafsampling.centerline import (
    compute_horizontal_centerline,
    compute_leaf_boundaries,
    compute_tangent_normal,
    trim_centerline_points,
)
from leafseg.io import make_rgb_8bit, read_image, save_png_rgb
from leafsampling.profiles import extract_green_profiles
from leafsampling.sampling import SamplingConfig, generate_sampling_lines, generate_strip_sampling_lines

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CenterlineSamplingConfig:
    """Configuration for geometric centerline and sampling-line generation."""

    min_leaf_width: int = 20
    tip_trim_ratio: float = 0.02
    smoothing_window_length: int = 51
    smoothing_polyorder: int = 2
    sampling_mode: str = "fixed_count"
    n_sampling_lines: int = 5
    sampling_step: int = 10
    strip_width_px: int = 5
    edge_trim_ratio: float = 0.05
    boundary_step_size: float = 0.5
    max_boundary_steps: int = 10000
    min_sampling_line_length: float = 20.0

    def sampling_config(self) -> SamplingConfig:
        """Return the subset of parameters used by sampling functions."""
        return SamplingConfig(
            sampling_mode=self.sampling_mode,
            n_sampling_lines=self.n_sampling_lines,
            sampling_step=self.sampling_step,
            strip_width_px=self.strip_width_px,
            edge_trim_ratio=self.edge_trim_ratio,
            boundary_step_size=self.boundary_step_size,
            max_boundary_steps=self.max_boundary_steps,
            min_sampling_line_length=self.min_sampling_line_length,
        )


def process_one_mask(
    mask_path: Path,
    output_dir: Path,
    config: CenterlineSamplingConfig,
    image_path: Path | None = None,
) -> dict[str, Any]:
    """Process one binary mask and save centerline, sampling-line, and QC outputs."""
    dirs = _ensure_centerline_output_dirs(output_dir)
    filename = mask_path.name
    output_stem = _output_stem(mask_path)
    try:
        mask = iio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_binary = mask > 0
        height, width = mask_binary.shape[:2]

        boundary_df = compute_leaf_boundaries(mask_binary, config.min_leaf_width)
        centerline_df = trim_centerline_points(boundary_df, config.tip_trim_ratio)
        if len(centerline_df) < 3:
            raise ValueError("Too few valid centerline points.")
        centerline_df = compute_horizontal_centerline(centerline_df)
        centerline_df = compute_tangent_normal(centerline_df)
        centerline_df.insert(0, "filename", filename)

        sampling_df = generate_sampling_lines(mask_binary, centerline_df, config.sampling_config())
        if not sampling_df.empty:
            sampling_df.insert(0, "filename", filename)

        centerline_df.to_csv(dirs["centerline"] / f"{output_stem}_centerline.csv", index=False)
        sampling_df.to_csv(dirs["sampling_lines"] / f"{output_stem}_sampling_lines.csv", index=False)

        profile_df = pd.DataFrame()
        if image_path is not None and image_path.exists() and not sampling_df.empty:
            original_image, _ = read_image(image_path)
            profile_df = extract_green_profiles(original_image, mask_binary, sampling_df, config.strip_width_px)
            if not profile_df.empty:
                profile_df.insert(0, "filename", filename)
        profile_df.to_csv(dirs["profiles"] / f"{output_stem}_green_profiles.csv", index=False)

        qc_flag, notes = _evaluate_centerline_qc(mask_binary, centerline_df, sampling_df, config)
        overlay_base = _load_overlay_base(image_path, mask_binary)
        overlay = create_centerline_sampling_overlay(overlay_base, mask_binary, centerline_df, sampling_df, filename)
        save_png_rgb(overlay, dirs["qc"] / f"{output_stem}_centerline_sampling_overlay.png")

        return {
            "filename": filename,
            "image_height": height,
            "image_width": width,
            "num_valid_centerline_points": len(centerline_df),
            "x_min_valid": float(centerline_df["x"].min()),
            "x_max_valid": float(centerline_df["x"].max()),
            "n_sampling_lines": int(sampling_df["sample_id"].nunique()) if not sampling_df.empty else 0,
            "smoothing_window_length": int(centerline_df.attrs.get("smoothing_window_length", 1)),
            "smoothing_polyorder": config.smoothing_polyorder,
            "strip_width_px": config.strip_width_px,
            "n_profile_rows": len(profile_df),
            "edge_trim_ratio": config.edge_trim_ratio,
            "qc_flag": qc_flag,
            "notes": notes,
        }
    except Exception as exc:
        LOGGER.warning("Failed to process mask %s: %s", mask_path, exc, exc_info=True)
        return _failure_metadata_row(filename, config, str(exc))


def process_mask_folder(
    mask_dir: Path,
    output_dir: Path,
    config: CenterlineSamplingConfig,
    image_dir: Path | None = None,
) -> None:
    """Process every mask PNG in a folder and save batch metadata."""
    dirs = _ensure_centerline_output_dirs(output_dir)
    mask_paths = sorted(mask_dir.glob("*.png"))
    LOGGER.info("Starting centerline/sampling batch")
    LOGGER.info("Found %d mask PNG files in %s", len(mask_paths), mask_dir)

    rows: list[dict[str, Any]] = []
    counts = {"pass": 0, "review": 0, "failed": 0}
    for index, mask_path in enumerate(mask_paths, start=1):
        LOGGER.info("Processing %d/%d: %s", index, len(mask_paths), mask_path.name)
        image_path = find_matching_image(mask_path, image_dir) if image_dir is not None else None
        row = process_one_mask(mask_path, output_dir, config, image_path)
        rows.append(row)
        flag = str(row.get("qc_flag", "failed"))
        counts[flag] = counts.get(flag, 0) + 1

    metadata_path = dirs["metadata"] / "centerline_sampling_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    LOGGER.info(
        "Centerline batch complete: pass=%d review=%d failed=%d",
        counts.get("pass", 0),
        counts.get("review", 0),
        counts.get("failed", 0),
    )


def find_matching_image(mask_path: Path, image_dir: Path) -> Path | None:
    """Find a likely crop image matching a crop mask filename."""
    stem = mask_path.stem
    candidate_stems = [
        stem.replace("_leaf_crop_mask", "_leaf_crop"),
        stem.replace("_leaf_mask", "_leaf_crop"),
        stem,
    ]
    extensions = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    for candidate_stem in candidate_stems:
        for extension in extensions:
            candidate = image_dir / f"{candidate_stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def create_centerline_sampling_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    centerline_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    filename: str,
) -> np.ndarray:
    """Draw mask boundary, geometric centerline, sampling lines, and sample centers."""
    overlay = image_rgb.copy()
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)

    if len(centerline_df) >= 2:
        points = centerline_df[["x", "y_center_smooth"]].to_numpy(dtype=np.float32).round().astype(np.int32)
        cv2.polylines(overlay, [points.reshape(-1, 1, 2)], isClosed=False, color=(255, 255, 0), thickness=2)

    if not sampling_df.empty:
        main_lines = sampling_df[sampling_df["strip_offset_px"] == 0] if "strip_offset_px" in sampling_df else sampling_df
        for _, row in main_lines.iterrows():
            pt1 = (int(round(row["x_start"])), int(round(row["y_start"])))
            pt2 = (int(round(row["x_end"])), int(round(row["y_end"])))
            center = (int(round(row["x_center"])), int(round(row["y_center"])))
            cv2.line(overlay, pt1, pt2, (180, 0, 255), thickness=1)
            cv2.circle(overlay, center, 2, (0, 255, 255), thickness=-1)

    _draw_label(overlay, filename, int(sampling_df["sample_id"].nunique()) if not sampling_df.empty else 0)
    return overlay


def _ensure_centerline_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "centerline": output_dir / "centerline",
        "sampling_lines": output_dir / "sampling_lines",
        "profiles": output_dir / "profiles",
        "qc": output_dir / "qc",
        "metadata": output_dir / "metadata",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _evaluate_centerline_qc(
    mask: np.ndarray,
    centerline_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    config: CenterlineSamplingConfig,
) -> tuple[str, str]:
    notes: list[str] = []
    if len(centerline_df) < max(10, config.smoothing_polyorder + 2):
        notes.append("too few valid centerline points")
    n_sampling = int(sampling_df["sample_id"].nunique()) if not sampling_df.empty else 0
    if n_sampling < max(3, int(config.n_sampling_lines * 0.5)) and config.sampling_mode == "fixed_count":
        notes.append("too few sampling lines generated")
    if not sampling_df.empty:
        short_ratio = float((sampling_df["line_length_px"] < config.min_sampling_line_length).mean())
        if short_ratio > 0.2:
            notes.append("many sampling lines have very short length")
    if _mask_touches_boundary(mask):
        notes.append("mask touches image boundary")
    return ("review" if notes else "pass", "; ".join(notes))


def _mask_touches_boundary(mask: np.ndarray) -> bool:
    mask_binary = np.asarray(mask) > 0
    if not np.any(mask_binary):
        return False
    return bool(
        np.any(mask_binary[0, :])
        or np.any(mask_binary[-1, :])
        or np.any(mask_binary[:, 0])
        or np.any(mask_binary[:, -1])
    )


def _load_overlay_base(image_path: Path | None, mask: np.ndarray) -> np.ndarray:
    if image_path is not None and image_path.exists():
        original, image_rgb = read_image(image_path)
        _ = original
        return image_rgb
    return make_rgb_8bit(np.where(mask > 0, 180, 0).astype(np.uint8))


def _draw_label(image_rgb: np.ndarray, filename: str, n_sampling_lines: int) -> None:
    text = f"{filename} | sampling lines: {n_sampling_lines}"
    cv2.rectangle(image_rgb, (0, 0), (min(image_rgb.shape[1] - 1, 900), 34), (255, 255, 255), thickness=-1)
    cv2.putText(image_rgb, text[:100], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def _output_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_leaf_crop_mask"):
        return stem[: -len("_leaf_crop_mask")]
    if stem.endswith("_leaf_mask"):
        return stem[: -len("_leaf_mask")]
    return stem


def _failure_metadata_row(filename: str, config: CenterlineSamplingConfig, note: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "image_height": 0,
        "image_width": 0,
        "num_valid_centerline_points": 0,
        "x_min_valid": "",
        "x_max_valid": "",
        "n_sampling_lines": 0,
        "smoothing_window_length": config.smoothing_window_length,
        "smoothing_polyorder": config.smoothing_polyorder,
        "strip_width_px": config.strip_width_px,
        "n_profile_rows": 0,
        "edge_trim_ratio": config.edge_trim_ratio,
        "qc_flag": "failed",
        "notes": note,
    }
