import numpy as np
import pandas as pd

from leafsampling.midrib import (
    MidribDetectionConfig,
    annotate_profile_sides,
    detect_midrib_peak_for_profile,
    summarize_midrib_detection,
)


def _profile(sample_id: int, peak_center: float = 0.46, width: float = 0.055, amplitude: float = 40.0) -> pd.DataFrame:
    x = np.linspace(0, 1, 201)
    y = 100 + amplitude * np.exp(-0.5 * ((x - peak_center) / width) ** 2)
    return pd.DataFrame(
        {
            "source_profile_file": "leaf_green_profiles.csv",
            "filename": "leaf_mask.png",
            "sample_id": sample_id,
            "position_fraction": x,
            "green_mean": y,
        }
    )


def test_detects_broad_middle_peak() -> None:
    result = detect_midrib_peak_for_profile(_profile(0), MidribDetectionConfig())

    assert result["status"] == "detected"
    assert 0.42 < result["peak_position_fraction"] < 0.50
    assert result["peak_width_fraction"] >= 0.03


def test_rejects_narrow_spike() -> None:
    x = np.linspace(0, 1, 201)
    y = np.full_like(x, 100.0)
    y[np.argmin(np.abs(x - 0.46))] = 180.0
    profile = pd.DataFrame({"sample_id": 0, "position_fraction": x, "green_mean": y})

    result = detect_midrib_peak_for_profile(
        profile,
        MidribDetectionConfig(smoothing_window_length=5, min_prominence=8.0, min_width_fraction=0.03),
    )

    assert result["status"] == "no_peak"


def test_leaf_summary_requires_consistent_peaks() -> None:
    rows = []
    for sample_id in range(5):
        result = detect_midrib_peak_for_profile(
            _profile(sample_id, peak_center=0.45 + 0.005 * sample_id),
            MidribDetectionConfig(),
        )
        result.update({"source_profile_file": "leaf_green_profiles.csv"})
        rows.append(result)
    summary = summarize_midrib_detection(pd.DataFrame(rows), MidribDetectionConfig())

    assert summary.loc[0, "qc_flag"] == "pass"
    assert summary.loc[0, "n_detected_peaks"] == 5


def test_annotates_upper_and_lower_sides() -> None:
    profiles = pd.concat([_profile(0)], ignore_index=True)
    line_results = pd.DataFrame(
        [
            {
                "source_profile_file": "leaf_green_profiles.csv",
                "sample_id": 0,
                "status": "detected",
                "peak_position_fraction": 0.45,
                "split_position_fraction": 0.45,
                "peak_left_fraction": 0.40,
                "peak_right_fraction": 0.50,
            }
        ]
    )

    annotated = annotate_profile_sides(profiles, line_results)

    assert "upper" in set(annotated["side_of_midrib"])
    assert "lower" in set(annotated["side_of_midrib"])


def test_detects_broad_middle_valley_with_dark_polarity() -> None:
    x = np.linspace(0, 1, 201)
    y = 140 - 45 * np.exp(-0.5 * ((x - 0.47) / 0.055) ** 2)
    profile = pd.DataFrame({"sample_id": 0, "position_fraction": x, "green_mean": y})

    result = detect_midrib_peak_for_profile(profile, MidribDetectionConfig(peak_polarity="dark"))

    assert result["status"] == "detected"
    assert 0.43 < result["peak_position_fraction"] < 0.51
