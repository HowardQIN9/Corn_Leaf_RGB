"""Mask-aware RGB image representations for leaf profile experiments."""

from __future__ import annotations

import numpy as np
from skimage import exposure


REPRESENTATIONS = ("raw_green", "ngrdi", "exg", "g2_rb", "clahe_green")


def calculate_representation(
    image: np.ndarray,
    mask: np.ndarray,
    representation: str,
    *,
    clahe_clip_limit: float = 0.01,
    clahe_grid_size: tuple[int, int] = (8, 8),
    denominator_pseudocount: float = 1.0 / 255.0,
) -> np.ndarray:
    """Return one scalar representation, with non-leaf pixels set to NaN.

    NGRDI, ExG, and G2/RB are calculated from RGB values normalized to [0, 1].
    The G2/RB denominator receives a fixed one-digital-count pseudocount per
    channel because the blue channel can contain zero-valued leaf pixels.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; choose from {REPRESENTATIONS}"
        )
    rgb = _rgb_float01(image)
    mask_binary = np.asarray(mask) > 0
    if rgb.shape[:2] != mask_binary.shape:
        raise ValueError(f"Image and mask shapes differ: {rgb.shape[:2]} vs {mask_binary.shape}")
    if not np.any(mask_binary):
        raise ValueError("Leaf mask is empty")

    red, green, blue = (rgb[:, :, index] for index in range(3))
    if representation == "raw_green":
        signal = _raw_green(image)
    elif representation == "ngrdi":
        signal = (green - red) / (green + red + denominator_pseudocount)
    elif representation == "exg":
        signal = 2.0 * green - red - blue
    elif representation == "g2_rb":
        signal = green**2 / (
            (red + denominator_pseudocount) * (blue + denominator_pseudocount)
        )
    else:
        signal = _masked_clahe(
            green,
            mask_binary,
            clip_limit=clahe_clip_limit,
            grid_size=clahe_grid_size,
        )

    output = np.asarray(signal, dtype=np.float32)
    output[~mask_binary] = np.nan
    if not np.isfinite(output[mask_binary]).all():
        raise FloatingPointError(f"Non-finite leaf pixels produced for {representation}")
    return output


def _rgb_float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        rgb = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported RGB image shape: {arr.shape}")

    rgb = rgb.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        scale = float(np.iinfo(arr.dtype).max)
    else:
        finite_max = float(np.nanmax(rgb)) if rgb.size else 1.0
        scale = 255.0 if finite_max > 1.0 else 1.0
    return np.clip(rgb / scale, 0.0, 1.0)


def _raw_green(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        return arr[:, :, 1].astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _masked_clahe(
    green01: np.ndarray,
    mask: np.ndarray,
    *,
    clip_limit: float,
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Apply fixed CLAHE without allowing black background to dominate.

    CLAHE itself has no mask argument. The tight leaf bounding box is used and
    non-leaf pixels are filled with the within-leaf median before enhancement;
    only enhanced pixels inside the original mask are returned to modeling.
    """
    rows, columns = np.where(mask)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    crop_green = np.clip(green01[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)
    crop_mask = mask[y0:y1, x0:x1]
    leaf_median = int(np.median(crop_green[crop_mask]))
    filled = crop_green.copy()
    filled[~crop_mask] = leaf_median
    kernel_size = (
        max(1, int(np.ceil(filled.shape[0] / int(grid_size[1])))),
        max(1, int(np.ceil(filled.shape[1] / int(grid_size[0])))),
    )
    enhanced_crop = exposure.equalize_adapthist(
        filled,
        kernel_size=kernel_size,
        clip_limit=float(clip_limit),
        nbins=256,
    ).astype(np.float32) * 255.0
    enhanced = np.full(mask.shape, np.nan, dtype=np.float32)
    enhanced[y0:y1, x0:x1][crop_mask] = enhanced_crop[crop_mask]
    return enhanced
