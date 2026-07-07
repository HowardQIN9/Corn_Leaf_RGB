"""Image input/output helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import tifffile


def list_image_files(input_dir: Path) -> list[Path]:
    """Return sorted supported image files in a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    patterns = (
        "*.tif",
        "*.tiff",
        "*.TIF",
        "*.TIFF",
        "*.jpg",
        "*.jpeg",
        "*.JPG",
        "*.JPEG",
        "*.png",
        "*.PNG",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))
    return sorted(set(files))


def read_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an image and return original pixels plus an 8-bit RGB preview."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        original = tifffile.imread(path)
    else:
        original = iio.imread(path)
    rgb_8bit = make_rgb_8bit(original)
    return original, rgb_8bit


def list_tiff_files(input_dir: Path) -> list[Path]:
    """Return sorted supported image files.

    Kept as a compatibility alias for the original TIFF-focused API.
    """
    return list_image_files(input_dir)


def make_rgb_8bit(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA image data to an 8-bit RGB preview."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            rgb = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported image channel count: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")
    return _scale_to_uint8(rgb)


def _scale_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    arr = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        scaled = arr / float(info.max)
    else:
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        scaled = arr / max_value if max_value > 1.0 else arr
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return standard output directories."""
    dirs = {
        "masks": output_dir / "masks",
        "crops": output_dir / "crops",
        "qc": output_dir / "qc",
        "metadata": output_dir / "metadata",
        "segmented_preview": output_dir / "segmented_preview",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a binary mask as PNG with values 0 and 255."""
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    iio.imwrite(path, binary)


def save_tiff(image: np.ndarray, path: Path) -> None:
    """Save image data as TIFF without changing dtype."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image)


def save_png_rgb(image_rgb: np.ndarray, path: Path) -> None:
    """Save an RGB image as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image_rgb.astype(np.uint8))


def make_segmented_preview(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create an RGB visualization with background set to white."""
    preview = image_rgb.copy()
    preview[mask == 0] = np.array([255, 255, 255], dtype=np.uint8)
    return preview


def encode_png_rgb(image_rgb: np.ndarray, path: Path) -> None:
    """Compatibility wrapper for saving RGB PNG images."""
    save_png_rgb(image_rgb, path)
