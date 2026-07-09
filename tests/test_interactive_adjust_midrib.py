from pathlib import Path

import pandas as pd

from scripts.interactive_adjust_midrib import find_profile_file, update_manual_split


def test_find_profile_file_requires_unique_match(tmp_path: Path) -> None:
    profile = tmp_path / "LeafDoc_212_tall_1_Leaf2_123_green_profiles.csv"
    profile.write_text("sample_id,position_fraction,green_mean\n0,0.5,100\n", encoding="utf-8")

    found = find_profile_file(tmp_path, "212_tall_1")

    assert found == profile


def test_update_manual_split_sets_use_manual(tmp_path: Path) -> None:
    manual_csv = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {
                "source_profile_file": "leaf_green_profiles.csv",
                "sample_id": 2,
                "manual_split_position_fraction": "",
                "use_manual": False,
                "manual_notes": "",
            }
        ]
    ).to_csv(manual_csv, index=False)

    update_manual_split(manual_csv, "leaf_green_profiles.csv", 2, 0.57, "clicked")
    updated = pd.read_csv(manual_csv)

    assert updated.loc[0, "manual_split_position_fraction"] == 0.57
    assert bool(updated.loc[0, "use_manual"]) is True
    assert updated.loc[0, "manual_notes"] == "clicked"
