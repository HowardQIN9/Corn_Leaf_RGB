"""Inspect candidate midrib peaks for one profile file."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    profiles = pd.read_csv("outputs/RGB_tall_v9_leaf2_centerline_sampling/metadata/all_leaf2_green_profiles.csv")
    case = profiles[profiles["source_profile_file"] == "LeafDoc_212_tall_1_Leaf2_1780600174352_green_profiles.csv"]
    for sample_id, profile in case.groupby("sample_id"):
        x = profile["position_fraction"].to_numpy(float)
        y = profile["green_mean"].to_numpy(float)
        window = 31
        if window >= len(y):
            window = len(y) - 1 if len(y) % 2 == 0 else len(y)
        y_smooth = savgol_filter(y, window_length=window, polyorder=2, mode="interp")
        mask = (x >= 0.35) & (x <= 0.70)
        idx = np.flatnonzero(mask)
        peaks, props = find_peaks(y_smooth[idx], prominence=3)
        widths = peak_widths(y_smooth[idx], peaks, rel_height=0.5)
        dx = float(np.median(np.diff(x)))
        rows = []
        for i, peak in enumerate(peaks):
            pos = x[idx[int(peak)]]
            width = float(widths[0][i] * dx)
            prom = float(props["prominences"][i])
            center_weight = float(np.exp(-0.5 * ((pos - 0.5) / 0.12) ** 2))
            rows.append((pos, prom, width, prom * width, prom * width * center_weight))
        print(f"\nsample_id={sample_id}")
        for row in sorted(rows, key=lambda item: item[-1], reverse=True)[:8]:
            print(
                f"pos={row[0]:.3f} prom={row[1]:.2f} width={row[2]:.3f} "
                f"score={row[3]:.3f} center_score={row[4]:.3f}"
            )


if __name__ == "__main__":
    main()
