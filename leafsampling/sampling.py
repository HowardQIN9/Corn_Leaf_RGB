"""Normal sampling line generation for geometric leaf centerlines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for centerline-normal sampling."""

    sampling_mode: str = "fixed_count"
    n_sampling_lines: int = 5
    sampling_step: int = 10
    strip_width_px: int = 5
    edge_trim_ratio: float = 0.05
    boundary_step_size: float = 0.5
    max_boundary_steps: int = 10000
    min_sampling_line_length: float = 20.0


def interpolate_centerline_geometry(centerline_df: pd.DataFrame, x: float) -> dict[str, float]:
    """Interpolate centerline y, tangent, and normal at a floating-point x position."""
    sorted_df = centerline_df.sort_values("x")
    x_values = sorted_df["x"].to_numpy(dtype=float)
    if x < x_values.min() or x > x_values.max():
        raise ValueError(f"x={x} is outside centerline range.")
    geometry: dict[str, float] = {"x": float(x)}
    for column in ("y_center_smooth", "tangent_x", "tangent_y", "normal_x", "normal_y", "dy_dx"):
        geometry[column] = float(np.interp(x, x_values, sorted_df[column].to_numpy(dtype=float)))

    tangent = _normalize(np.array([geometry["tangent_x"], geometry["tangent_y"]], dtype=float))
    normal = _normalize(np.array([geometry["normal_x"], geometry["normal_y"]], dtype=float))
    geometry["tangent_x"] = float(tangent[0])
    geometry["tangent_y"] = float(tangent[1])
    geometry["normal_x"] = float(normal[0])
    geometry["normal_y"] = float(normal[1])
    return geometry


def march_to_mask_boundary(
    mask: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    step_size: float = 0.5,
    max_steps: int = 10000,
) -> np.ndarray:
    """March from center along direction and return the last inside-mask point."""
    mask_binary = np.asarray(mask) > 0
    direction = _normalize(direction)
    current = center.astype(float).copy()
    if not _point_inside_mask(mask_binary, current):
        return current
    last_inside = current.copy()
    for _ in range(max_steps):
        current = current + direction * step_size
        if not _point_inside_mask(mask_binary, current):
            break
        last_inside = current.copy()
    return last_inside


def generate_normal_sampling_line(
    mask: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    edge_trim_ratio: float,
    step_size: float = 0.5,
    max_steps: int = 10000,
) -> dict[str, float] | None:
    """Generate one normal line endpoint pair inside the mask."""
    mask_binary = np.asarray(mask) > 0
    center = center.astype(float)
    normal = _normalize(normal)
    if not _point_inside_mask(mask_binary, center):
        return None

    positive = march_to_mask_boundary(mask_binary, center, normal, step_size, max_steps)
    negative = march_to_mask_boundary(mask_binary, center, -normal, step_size, max_steps)

    p_start = negative
    p_end = positive
    vector = p_end - p_start
    p_start_trimmed = p_start + edge_trim_ratio * vector
    p_end_trimmed = p_end - edge_trim_ratio * vector
    line_length = float(np.linalg.norm(p_end_trimmed - p_start_trimmed))
    if line_length <= 0:
        return None

    return {
        "x_start": float(p_start_trimmed[0]),
        "y_start": float(p_start_trimmed[1]),
        "x_end": float(p_end_trimmed[0]),
        "y_end": float(p_end_trimmed[1]),
        "line_length_px": line_length,
    }


def generate_sampling_lines(mask: np.ndarray, centerline_df: pd.DataFrame, config: SamplingConfig) -> pd.DataFrame:
    """Generate one normal sampling line per sampling x position."""
    rows: list[dict[str, float | int | str]] = []
    for sample_id, x in enumerate(_sampling_x_positions(centerline_df, config)):
        geometry = interpolate_centerline_geometry(centerline_df, float(x))
        center = np.array([x, geometry["y_center_smooth"]], dtype=float)
        normal = np.array([geometry["normal_x"], geometry["normal_y"]], dtype=float)
        line = generate_normal_sampling_line(
            mask,
            center,
            normal,
            config.edge_trim_ratio,
            config.boundary_step_size,
            config.max_boundary_steps,
        )
        if line is None:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "x_center": float(center[0]),
                "y_center": float(center[1]),
                "tangent_x": geometry["tangent_x"],
                "tangent_y": geometry["tangent_y"],
                "normal_x": geometry["normal_x"],
                "normal_y": geometry["normal_y"],
                **line,
                "strip_width_px": config.strip_width_px,
                "edge_trim_ratio": config.edge_trim_ratio,
                "coordinate_system": "image_xy_y_down",
            }
        )
    return pd.DataFrame(rows)


def generate_strip_sampling_lines(mask: np.ndarray, centerline_df: pd.DataFrame, config: SamplingConfig) -> pd.DataFrame:
    """Generate strip sub-lines by shifting each sampling center along local tangent."""
    rows: list[dict[str, float | int | str]] = []
    offsets = _strip_offsets(config.strip_width_px)
    for sample_id, x in enumerate(_sampling_x_positions(centerline_df, config)):
        geometry = interpolate_centerline_geometry(centerline_df, float(x))
        base_center = np.array([x, geometry["y_center_smooth"]], dtype=float)
        tangent = np.array([geometry["tangent_x"], geometry["tangent_y"]], dtype=float)
        normal = np.array([geometry["normal_x"], geometry["normal_y"]], dtype=float)
        for offset in offsets:
            shifted_center = base_center + tangent * float(offset)
            line = generate_normal_sampling_line(
                mask,
                shifted_center,
                normal,
                config.edge_trim_ratio,
                config.boundary_step_size,
                config.max_boundary_steps,
            )
            if line is None:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "strip_offset_px": int(offset),
                    "x_center": float(base_center[0]),
                    "y_center": float(base_center[1]),
                    "x_center_shifted": float(shifted_center[0]),
                    "y_center_shifted": float(shifted_center[1]),
                    "tangent_x": geometry["tangent_x"],
                    "tangent_y": geometry["tangent_y"],
                    "normal_x": geometry["normal_x"],
                    "normal_y": geometry["normal_y"],
                    **line,
                    "strip_width_px": config.strip_width_px,
                    "edge_trim_ratio": config.edge_trim_ratio,
                    "coordinate_system": "image_xy_y_down",
                }
            )
    return pd.DataFrame(rows)


def _sampling_x_positions(centerline_df: pd.DataFrame, config: SamplingConfig) -> np.ndarray:
    x_min = float(centerline_df["x"].min())
    x_max = float(centerline_df["x"].max())
    if config.sampling_mode == "fixed_count":
        n_lines = max(0, int(config.n_sampling_lines))
        if n_lines == 0:
            return np.array([], dtype=float)
        # Use interior positions so the requested five-point sampling avoids leaf tips.
        fractions = np.arange(1, n_lines + 1, dtype=float) / float(n_lines + 1)
        return x_min + fractions * (x_max - x_min)
    if config.sampling_mode == "fixed_step":
        step = max(1, int(config.sampling_step))
        return np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    raise ValueError(f"Unsupported sampling mode: {config.sampling_mode}")


def _strip_offsets(strip_width_px: int) -> list[int]:
    width = max(1, int(strip_width_px))
    half = width // 2
    if width % 2 == 1:
        return list(range(-half, half + 1))
    return list(range(-half, half))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def _point_inside_mask(mask_binary: np.ndarray, point_xy: np.ndarray) -> bool:
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    if y < 0 or x < 0 or y >= mask_binary.shape[0] or x >= mask_binary.shape[1]:
        return False
    return bool(mask_binary[y, x])
