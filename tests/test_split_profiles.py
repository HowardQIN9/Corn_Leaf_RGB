import pandas as pd

from leafsampling.split_profiles import split_profiles_from_midrib_region


def test_split_profiles_start_from_midrib_boundary() -> None:
    profiles = pd.DataFrame(
        {
            "source_profile_file": ["leaf.csv"] * 11,
            "sample_id": [0] * 11,
            "position_fraction": [i / 10 for i in range(11)],
            "green_mean": list(range(11)),
        }
    )
    line_results = pd.DataFrame(
        {
            "source_profile_file": ["leaf.csv"],
            "sample_id": [0],
            "peak_left_fraction": [0.4],
            "peak_right_fraction": [0.6],
        }
    )

    split = split_profiles_from_midrib_region(profiles, line_results)
    upper = split[split["midrib_side"] == "upper"]
    lower = split[split["midrib_side"] == "lower"]

    assert upper.iloc[0]["position_fraction"] == 0.4
    assert lower.iloc[0]["position_fraction"] == 0.6
    assert upper.iloc[0]["relative_distance_from_midrib"] == 0.0
    assert lower.iloc[0]["relative_distance_from_midrib"] == 0.0
    assert set(split["midrib_side"]) == {"upper", "lower"}
