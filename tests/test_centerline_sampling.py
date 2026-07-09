import cv2
import numpy as np

from leafsampling.centerline import (
    compute_horizontal_centerline,
    compute_leaf_boundaries,
    compute_tangent_normal,
    trim_centerline_points,
)
from leafsampling.sampling import (
    SamplingConfig,
    generate_normal_sampling_line,
    generate_strip_sampling_lines,
)
from leafsampling.profiles import extract_green_profiles


def _ellipse_mask() -> np.ndarray:
    mask = np.zeros((120, 240), dtype=np.uint8)
    cv2.ellipse(mask, center=(120, 60), axes=(95, 28), angle=5, startAngle=0, endAngle=360, color=255, thickness=-1)
    return mask


def _centerline_df(mask: np.ndarray):
    df = compute_leaf_boundaries(mask, min_leaf_width=10)
    df = trim_centerline_points(df, trim_ratio=0.02)
    df = compute_horizontal_centerline(df)
    return compute_tangent_normal(df)


def test_centerline_extraction_from_synthetic_ellipse_mask() -> None:
    mask = _ellipse_mask()

    df = _centerline_df(mask)

    assert len(df) > 100
    assert df["y_center_smooth"].nunique() == 1
    assert df["y_center_smooth"].between(45, 75).all()
    assert {"y_top", "y_bottom", "leaf_width", "y_center_raw", "y_center_smooth"}.issubset(df.columns)


def test_tangent_and_normal_are_normalized() -> None:
    df = _centerline_df(_ellipse_mask())

    tangent_norm = np.sqrt(df["tangent_x"] ** 2 + df["tangent_y"] ** 2)
    normal_norm = np.sqrt(df["normal_x"] ** 2 + df["normal_y"] ** 2)

    assert np.allclose(tangent_norm, 1.0)
    assert np.allclose(normal_norm, 1.0)


def test_sampling_line_endpoints_inside_mask() -> None:
    mask = _ellipse_mask()
    df = _centerline_df(mask)
    center_row = df.iloc[len(df) // 2]
    center = np.array([center_row["x"], center_row["y_center_smooth"]], dtype=float)
    normal = np.array([center_row["normal_x"], center_row["normal_y"]], dtype=float)

    line = generate_normal_sampling_line(mask, center, normal, edge_trim_ratio=0.05)

    assert line is not None
    for x_name, y_name in (("x_start", "y_start"), ("x_end", "y_end")):
        x = int(round(line[x_name]))
        y = int(round(line[y_name]))
        assert mask[y, x] > 0


def test_strip_width_five_creates_five_sub_lines_per_sampling_location() -> None:
    mask = _ellipse_mask()
    df = _centerline_df(mask)
    config = SamplingConfig(n_sampling_lines=10, strip_width_px=5)

    lines = generate_strip_sampling_lines(mask, df, config)

    counts = lines.groupby("sample_id").size()
    assert len(counts) == 10
    assert (counts == 5).all()
    assert set(lines["strip_offset_px"].unique()) == {-2, -1, 0, 1, 2}


def test_default_fixed_count_creates_five_main_sampling_lines() -> None:
    mask = _ellipse_mask()
    df = _centerline_df(mask)
    config = SamplingConfig()

    from leafsampling.sampling import generate_sampling_lines

    lines = generate_sampling_lines(mask, df, config)

    assert len(lines) == 5
    assert sorted(lines["sample_id"].unique()) == [0, 1, 2, 3, 4]


def test_green_profile_uses_five_pixel_average() -> None:
    mask = _ellipse_mask()
    df = _centerline_df(mask)
    from leafsampling.sampling import generate_sampling_lines

    lines = generate_sampling_lines(mask, df, SamplingConfig())
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for x in range(mask.shape[1]):
        image[:, x, 1] = x

    profiles = extract_green_profiles(image, mask, lines.iloc[[2]], profile_width_px=5)

    assert not profiles.empty
    assert profiles["n_pixels_averaged"].max() == 5
    assert {"sample_id", "position_index", "green_mean", "profile_width_px"}.issubset(profiles.columns)
