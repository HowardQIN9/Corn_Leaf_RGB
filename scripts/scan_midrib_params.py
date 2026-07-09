"""Quick parameter scan for midrib valley/peak detection thresholds."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.midrib import MidribDetectionConfig, detect_midrib_peaks, summarize_midrib_detection


def main() -> None:
    profiles = pd.read_csv("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv")
    for prominence in [2, 3, 5, 8, 12]:
        for width in [0.005, 0.01, 0.02, 0.03, 0.05]:
            config = MidribDetectionConfig(
                peak_polarity="dark",
                min_prominence=prominence,
                min_width_fraction=width,
                max_peak_position_range=0.12,
            )
            line_results = detect_midrib_peaks(profiles, config)
            summary = summarize_midrib_detection(line_results, config)
            n_lines = int((line_results["status"] == "detected").sum())
            n_pass = int((summary["qc_flag"] == "pass").sum())
            print(
                f"prom={prominence:>2} width={width:>5}: "
                f"detected_lines={n_lines:>3}/525 pass_leaf={n_pass:>3}/105"
            )
    print("\nConsistency range scan for prom=3 width=0.005")
    for peak_range in [0.08, 0.12, 0.16, 0.20, 0.25, 0.30]:
        config = MidribDetectionConfig(
            peak_polarity="dark",
            min_prominence=3,
            min_width_fraction=0.005,
            max_peak_position_range=peak_range,
        )
        line_results = detect_midrib_peaks(profiles, config)
        summary = summarize_midrib_detection(line_results, config)
        n_pass = int((summary["qc_flag"] == "pass").sum())
        print(f"max_peak_position_range={peak_range:>4}: pass_leaf={n_pass:>3}/105")


if __name__ == "__main__":
    main()
