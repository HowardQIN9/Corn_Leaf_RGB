import pandas as pd

from scripts.manual_adjust_midrib import _merge_manual_adjustments


def test_manual_split_overrides_auto_split() -> None:
    line_results = pd.DataFrame(
        [
            {
                "source_profile_file": "leaf.csv",
                "sample_id": 0,
                "status": "detected",
                "peak_position_fraction": 0.50,
                "split_position_fraction": 0.50,
                "peak_left_fraction": 0.45,
                "peak_right_fraction": 0.55,
            }
        ]
    )
    manual = pd.DataFrame(
        [
            {
                "source_profile_file": "leaf.csv",
                "sample_id": 0,
                "manual_split_position_fraction": 0.62,
                "use_manual": True,
                "manual_notes": "visual correction",
            }
        ]
    )

    adjusted = _merge_manual_adjustments(line_results, manual)

    assert adjusted.loc[0, "split_position_fraction"] == 0.62
    assert adjusted.loc[0, "split_source"] == "manual"
    assert adjusted.loc[0, "status"] == "detected"
    assert adjusted.loc[0, "peak_left_fraction"] == 0.57
    assert adjusted.loc[0, "peak_right_fraction"] == 0.67
