import numpy as np
import pandas as pd

from leafsampling.curve_separation import create_valley_envelope, separate_profile_components


def test_create_valley_envelope_matches_profile_length() -> None:
    y_raw = np.array([5.0, 7.0, 5.5, 12.0, 6.0, 8.0, 5.0])

    envelope = create_valley_envelope(y_raw, valley_distance=2, smooth_window=5)

    assert len(envelope) == len(y_raw)
    assert np.isfinite(envelope).all()


def test_separate_profile_components_adds_meso_and_peak_per_side() -> None:
    split_profiles = pd.DataFrame(
        {
            "source_profile_file": ["leaf.csv"] * 6,
            "sample_id": [0] * 6,
            "midrib_side": ["upper"] * 3 + ["lower"] * 3,
            "distance_index": [0, 1, 2, 0, 1, 2],
            "relative_distance_from_midrib": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "green_mean": [10.0, 15.0, 11.0, 20.0, 30.0, 21.0],
        }
    )

    separated = separate_profile_components(split_profiles, valley_distance=2, smooth_window=5)

    assert {"green_mean_meso", "green_mean_peak"} <= set(separated.columns)
    np.testing.assert_allclose(
        separated["green_mean_peak"],
        separated["green_mean"] - separated["green_mean_meso"],
    )
    assert separated.groupby(["sample_id", "midrib_side"]).size().to_dict() == {(0, "lower"): 3, (0, "upper"): 3}
