import numpy as np
import pandas as pd
import pytest

from leafsampling.features import (
    aggregate_image_features,
    extract_curve_features,
    signal_features,
    zero_crossing_rate,
)


def _profiles() -> pd.DataFrame:
    rows = []
    for side, offset in (("upper", 0.0), ("lower", 2.0)):
        x = np.linspace(0.0, 1.0, 21)
        meso = 10.0 + offset + 3.0 * x
        peak = np.exp(-((x - 0.5) ** 2) / 0.01) * 4.0
        raw = meso + peak
        for index, (distance, raw_value, meso_value, peak_value) in enumerate(
            zip(x, raw, meso, peak, strict=True)
        ):
            rows.append(
                {
                    "prefix": "LeafDoc",
                    "plot_number": 109,
                    "geno": "tall",
                    "plant_number": 1,
                    "leaf": "Leaf2",
                    "timestamp": 123,
                    "source_profile_file": "leaf.csv",
                    "filename": "leaf_mask.png",
                    "sample_id": 0,
                    "midrib_side": side,
                    "distance_index": index,
                    "relative_distance_from_midrib": distance,
                    "green_mean": raw_value,
                    "green_mean_meso": meso_value,
                    "green_mean_peak": peak_value,
                }
            )
    return pd.DataFrame(rows)


def test_zero_crossing_rate_ignores_exact_zeros() -> None:
    assert zero_crossing_rate([-1.0, 0.0, 1.0, 0.0, -1.0]) == pytest.approx(0.5)
    assert zero_crossing_rate([0.0, 0.0, 0.0]) == 0.0


def test_signal_features_recover_linear_slope() -> None:
    x = np.linspace(0.0, 1.0, 20)
    features = signal_features(2.0 + 3.0 * x, "test", x=x)

    assert features["test_linear_slope"] == pytest.approx(3.0)
    assert features["test_zero_cross_rate"] == 0.0
    assert features["test_n_points"] == 20.0


def test_extract_curve_features_creates_rows_and_zones() -> None:
    features = extract_curve_features(_profiles(), resample_points=31, n_zones=3)

    assert len(features) == 2
    assert set(features["midrib_side"]) == {"upper", "lower"}
    assert "green_mean__full__signal_mean" in features.columns
    assert "green_mean_meso__proximal__d1_mean" in features.columns
    assert "green_mean_meso__middle__curvature_mean" in features.columns
    assert "green_mean_peak__distal__peaks_count" in features.columns
    assert np.isfinite(features["green_mean_meso__full__d1_mean"]).all()
    np.testing.assert_allclose(features["green_mean_meso__full__d1_mean"], 3.0, atol=1e-10)


def test_aggregate_image_features_returns_one_image_row() -> None:
    curve_features = extract_curve_features(_profiles(), resample_points=31, n_zones=1)
    image_features = aggregate_image_features(curve_features, aggregations=("mean", "std"))

    assert len(image_features) == 1
    assert image_features.iloc[0]["n_curves"] == 2
    assert "all_curves__green_mean__full__signal_mean__mean" in image_features.columns
    assert image_features.iloc[0]["plot_number"] == 109


def test_extract_curve_features_validates_required_columns() -> None:
    with pytest.raises(ValueError, match="green_mean_peak"):
        extract_curve_features(_profiles().drop(columns="green_mean_peak"))


def test_derivative_column_does_not_have_to_be_a_signal_output() -> None:
    features = extract_curve_features(
        _profiles(),
        value_columns=("green_mean",),
        derivative_column="green_mean_meso",
        resample_points=31,
    )

    assert "green_mean__full__signal_mean" in features.columns
    assert "green_mean_meso__full__d1_mean" in features.columns
    assert "green_mean_meso__full__signal_mean" not in features.columns
