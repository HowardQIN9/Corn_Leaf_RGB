from __future__ import annotations

import numpy as np

from leafsampling.profiles import extract_scalar_profiles
from leafsampling.representations import REPRESENTATIONS, calculate_representation


def test_representations_are_finite_only_inside_mask() -> None:
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = 100
    image[:, :, 2] = 20
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[2:10, 3:12] = 255

    for name in REPRESENTATIONS:
        result = calculate_representation(image, mask, name)
        assert result.shape == mask.shape
        assert np.isfinite(result[mask > 0]).all()
        assert np.isnan(result[mask == 0]).all()


def test_index_formulas_use_normalized_rgb_and_pseudocount() -> None:
    image = np.array([[[51, 102, 0]]], dtype=np.uint8)
    mask = np.ones((1, 1), dtype=np.uint8)
    eps = 1.0 / 255.0
    r, g, b = 51 / 255, 102 / 255, 0.0

    ngrdi = calculate_representation(image, mask, "ngrdi")[0, 0]
    exg = calculate_representation(image, mask, "exg")[0, 0]
    g2_rb = calculate_representation(image, mask, "g2_rb")[0, 0]
    assert np.isclose(ngrdi, (g - r) / (g + r + eps))
    assert np.isclose(exg, 2 * g - r - b)
    assert np.isclose(g2_rb, g**2 / ((r + eps) * (b + eps)))


def test_scalar_profiles_ignore_nonfinite_and_outside_mask() -> None:
    values = np.arange(25, dtype=float).reshape(5, 5)
    values[2, 2] = np.nan
    mask = np.ones((5, 5), dtype=np.uint8)
    mask[:, 0] = 0
    sampling = np.array(
        [(0, 0.0, 2.0, 4.0, 2.0, 0.0, 1.0)],
        dtype=[
            ("sample_id", "i4"),
            ("x_start", "f8"),
            ("y_start", "f8"),
            ("x_end", "f8"),
            ("y_end", "f8"),
            ("tangent_x", "f8"),
            ("tangent_y", "f8"),
        ],
    )
    import pandas as pd

    profiles = extract_scalar_profiles(
        values,
        mask,
        pd.DataFrame(sampling),
        profile_width_px=1,
        value_name="signal",
    )
    assert len(profiles) == 3
    assert profiles["signal_mean"].notna().all()
