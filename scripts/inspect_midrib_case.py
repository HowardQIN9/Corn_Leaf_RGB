"""Inspect midrib detection parameters for one profile file."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.midrib import MidribDetectionConfig, detect_midrib_peaks


def main() -> None:
    profiles = pd.read_csv("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv")
    case = profiles[profiles["source_profile_file"] == "LeafDoc_212_tall_1_Leaf2_1780600174352_green_profiles.csv"]
    for width in [0.003, 0.005, 0.008, 0.01, 0.015]:
        config = MidribDetectionConfig(
            peak_polarity="bright",
            middle_min_fraction=0.35,
            middle_max_fraction=0.70,
            min_prominence=6,
            min_width_fraction=width,
            max_peak_position_range=0.25,
        )
        line_results = detect_midrib_peaks(case, config)
        print(f"\nwidth={width}")
        print(
            line_results[
                [
                    "sample_id",
                    "status",
                    "peak_position_fraction",
                    "split_position_fraction",
                    "peak_prominence",
                    "peak_width_fraction",
                    "notes",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
