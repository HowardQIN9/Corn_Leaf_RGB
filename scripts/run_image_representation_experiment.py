"""Run five separate mask-only image-representation classification pipelines.

The saved masks, sampling lines, and manually adjusted midrib boundaries are
held fixed. Each representation is processed independently through profile
sampling, curve feature extraction, and three-class leave-one-block-out
classification. Features from different representations are never combined.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leafsampling.curve_separation import separate_profile_components
from leafsampling.features import extract_curve_features
from leafsampling.profiles import extract_scalar_profiles
from leafsampling.representations import REPRESENTATIONS, calculate_representation
from leafsampling.split_profiles import split_profiles_from_midrib_region
from scripts.extract_leaf2_crops import parse_leaf_filename


DEFAULT_CROP_DIR = Path("outputs/RGB_tall_v9_leaf2_crops")
DEFAULT_SAMPLING_DIR = Path("outputs/RGB_tall_v9_leaf2_centerline_sampling/sampling_lines")
DEFAULT_MIDRIB_RESULTS = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/"
    "midrib_peak_line_results_manual_adjusted.csv"
)
DEFAULT_LABELS = Path(
    r"C:\Users\98748\OneDrive - purdue.edu\PlantSensorLab\2026Summer\n_rate_plot_st_corrected.xlsx"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/image_representation_experiment"
)

VALUE_COLUMNS = ("signal_mean", "signal_mean_meso", "signal_mean_peak")
REPRESENTATION_LABELS = {
    "raw_green": "Raw Green",
    "ngrdi": "NGRDI",
    "exg": "ExG",
    "g2_rb": "G2/(R*B)",
    "clahe_green": "CLAHE Green",
}
REPRESENTATION_FORMULAS = {
    "raw_green": "G in original 8-bit units",
    "ngrdi": "(G-R)/(G+R+1/255)",
    "exg": "2G-R-B",
    "g2_rb": "G^2/((R+1/255)(B+1/255))",
    "clahe_green": "mask-aware G CLAHE, normalized clipLimit=0.01, tileGridSize=8x8",
}


def build_curve_features(
    representation: str,
    crop_dir: Path,
    sampling_dir: Path,
    midrib_results_csv: Path,
    output_csv: Path,
    qc_dir: Path,
    *,
    qc_examples: int = 3,
    clahe_clip_limit: float = 0.01,
    clahe_grid_size: tuple[int, int] = (8, 8),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one compact curve-feature table for a representation."""
    crop_paths = sorted(crop_dir.glob("*_leaf_crop.tif"))
    if not crop_paths:
        raise FileNotFoundError(f"No leaf crop TIFFs found in {crop_dir}")
    line_results = pd.read_csv(midrib_results_csv)
    feature_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    qc_dir.mkdir(parents=True, exist_ok=True)

    for index, crop_path in enumerate(crop_paths, start=1):
        base = crop_path.stem.removesuffix("_leaf_crop")
        mask_path = crop_dir / f"{base}_leaf_crop_mask.png"
        sampling_path = sampling_dir / f"{base}_sampling_lines.csv"
        source_profile_file = f"{base}_green_profiles.csv"
        if not mask_path.exists() or not sampling_path.exists():
            raise FileNotFoundError(
                f"Missing mask or sampling geometry for {crop_path.name}: "
                f"{mask_path.name}, {sampling_path.name}"
            )

        image = tifffile.imread(crop_path)
        mask = iio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_binary = mask > 0
        sampling = pd.read_csv(sampling_path)
        scalar = calculate_representation(
            image,
            mask_binary,
            representation,
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )
        profiles = extract_scalar_profiles(
            scalar,
            mask_binary,
            sampling,
            profile_width_px=5,
            value_name="signal",
        )

        info = parse_leaf_filename(crop_path)
        if info is None:
            raise ValueError(f"Cannot parse leaf filename: {crop_path.name}")
        metadata = {
            "prefix": info.prefix,
            "plot_number": int(info.plot_number),
            "geno": info.geno,
            "plant_number": int(info.plant_number),
            "leaf": info.leaf,
            "timestamp": info.timestamp,
            "source_profile_file": source_profile_file,
            "filename": mask_path.name,
        }
        for column, value in reversed(list(metadata.items())):
            profiles.insert(0, column, value)

        leaf_lines = line_results[line_results["source_profile_file"] == source_profile_file]
        if leaf_lines["sample_id"].nunique() != 5:
            raise AssertionError(
                f"Expected five fixed midrib line results for {source_profile_file}, "
                f"found {leaf_lines['sample_id'].nunique()}"
            )
        split = split_profiles_from_midrib_region(profiles, leaf_lines)
        separated = separate_profile_components(
            split,
            value_column="signal_mean",
            valley_distance=7,
            smooth_window=21,
        )
        features = extract_curve_features(
            separated,
            value_columns=VALUE_COLUMNS,
            derivative_column="signal_mean_meso",
            n_zones=3,
            resample_points=256,
            min_points=5,
            peak_prominence_fraction=0.10,
        )
        if len(features) != 10:
            raise AssertionError(
                f"Expected 10 curves for {source_profile_file}, found {len(features)}"
            )
        feature_tables.append(features)

        leaf_values = scalar[mask_binary]
        summaries.append(
            {
                "representation": representation,
                "source_profile_file": source_profile_file,
                "plot_number": int(info.plot_number),
                "plant_number": int(info.plant_number),
                "leaf_min": float(np.min(leaf_values)),
                "leaf_q01": float(np.percentile(leaf_values, 1)),
                "leaf_median": float(np.median(leaf_values)),
                "leaf_q99": float(np.percentile(leaf_values, 99)),
                "leaf_max": float(np.max(leaf_values)),
            }
        )
        if index <= qc_examples:
            _save_qc_preview(scalar, mask_binary, qc_dir / f"{base}_{representation}.png")
        if index == 1 or index % 20 == 0 or index == len(crop_paths):
            print(f"  {representation}: processed {index}/{len(crop_paths)} leaves", flush=True)

    curve_features = pd.concat(feature_tables, ignore_index=True)
    if len(curve_features) != len(crop_paths) * 10:
        raise AssertionError("Curve-feature row count does not equal leaves x 10 curves")
    if curve_features["source_profile_file"].nunique() != len(crop_paths):
        raise AssertionError("Source leaf count changed during feature extraction")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_features.to_csv(output_csv, index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_csv.parent / "image_signal_summary.csv", index=False)
    return curve_features, summary


def run_classification(
    feature_csv: Path,
    labels: Path,
    output_dir: Path,
) -> None:
    """Run the existing leakage-safe three-class classifier as a subprocess."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("select_features_classify_treatment.py")),
        "--input",
        str(feature_csv),
        "--labels",
        str(labels),
        "--output_dir",
        str(output_dir),
        "--three_class",
    ]
    subprocess.run(command, check=True)


def summarize_experiment(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine independent model metrics and feature-stability summaries."""
    rows: list[pd.DataFrame] = []
    for representation in REPRESENTATIONS:
        classification_dir = output_root / representation / "treatment_classification_3class"
        metrics = pd.read_csv(classification_dir / "oof_metrics.csv")
        frequency = pd.read_csv(classification_dir / "feature_selection_frequency.csv")
        metrics.insert(0, "representation", representation)
        metrics.insert(1, "representation_label", REPRESENTATION_LABELS[representation])
        metrics["chance_accuracy"] = 1.0 / 3.0
        metrics["above_chance_accuracy"] = metrics["accuracy"] - 1.0 / 3.0
        metrics["max_feature_folds_selected"] = int(frequency["folds_selected"].max())
        metrics["features_selected_all_folds"] = int((frequency["folds_selected"] == 3).sum())
        rows.append(metrics)
    combined = pd.concat(rows, ignore_index=True)
    best = (
        combined.sort_values(
            ["representation", "balanced_accuracy", "macro_f1"],
            ascending=[True, False, False],
        )
        .groupby("representation", sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(comparison_dir / "representation_model_metrics.csv", index=False)
    best.to_csv(comparison_dir / "representation_best_models.csv", index=False)
    wide = combined.pivot(
        index=["representation", "representation_label"],
        columns="model",
        values=["accuracy", "balanced_accuracy", "macro_f1"],
    )
    wide.columns = [f"{model}__{metric}" for metric, model in wide.columns]
    wide = wide.reset_index()
    representation_order = {name: index for index, name in enumerate(REPRESENTATIONS)}
    wide["_display_order"] = wide["representation"].map(representation_order)
    wide = wide.sort_values("_display_order").drop(columns="_display_order")
    ordered_metric_columns = [
        f"{model}__{metric}"
        for model in ("logistic_l2", "nearest_centroid", "pls_da")
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    ]
    wide = wide[["representation", "representation_label", *ordered_metric_columns]]
    wide.to_csv(comparison_dir / "representation_metrics_wide.csv", index=False)

    controlled_variables = pd.DataFrame(
        [
            ("Independent unit", "Plot", "21 plots"),
            ("Class grouping", "Fixed", "Low={0,60}; Medium={85,120,153}; High={180,240}"),
            ("Cross-validation", "Fixed", "Leave-one-block-out; identical three held-out blocks"),
            ("Leaf mask", "Fixed", "Same saved mask for every representation"),
            ("Sampling geometry", "Fixed", "Same five transverse lines per leaf"),
            ("Midrib boundaries", "Fixed", "Same manually adjusted boundaries"),
            ("Curve aggregation", "Fixed", "10 curves -> leaf median; 5 leaves -> plot median"),
            ("Curve feature algorithm", "Fixed", "Same 696 candidate features"),
            ("Feature selection", "Fixed method", "Train-fold-only filtering, correlation, ANOVA Top 20, mRMR Top 3"),
            ("Classifier settings", "Fixed", "Same Logistic L2, Nearest Centroid, and <=2-component PLS-DA"),
            ("Changed variable", "Image representation only", "Raw Green, NGRDI, ExG, G2/(R*B), or CLAHE Green"),
            ("Feature combination", "Not allowed", "Representations modeled separately"),
        ],
        columns=["comparison_item", "status", "implementation"],
    )
    controlled_variables.to_csv(comparison_dir / "controlled_variables.csv", index=False)
    raw_plot_table = pd.read_csv(
        output_root / "raw_green" / "treatment_classification_3class" / "plot_level_median_features.csv"
    )
    mapping_columns = [
        column for column in ("plot_number", "block", "n_rate", "treatment")
        if column in raw_plot_table.columns
    ]
    raw_plot_table[mapping_columns].drop_duplicates().sort_values("plot_number").to_csv(
        comparison_dir / "plot_treatment_mapping_used.csv", index=False
    )
    _plot_metric_comparison(combined, comparison_dir / "representation_balanced_accuracy.png")
    return combined, best


def _save_qc_preview(values: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    """Save a percentile-scaled preview; scaling is visualization-only."""
    leaf_values = values[mask]
    lower, upper = np.percentile(leaf_values, [1, 99])
    if upper <= lower:
        upper = lower + 1.0
    preview = np.zeros(values.shape, dtype=np.uint8)
    preview[mask] = np.clip((values[mask] - lower) / (upper - lower) * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(output_path, preview)


def _plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    models = ["logistic_l2", "nearest_centroid", "pls_da"]
    labels = [REPRESENTATION_LABELS[name] for name in REPRESENTATIONS]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 5.8))
    colors = ["#32688e", "#58a65c", "#d07c31"]
    for model_index, (model, color) in enumerate(zip(models, colors, strict=True)):
        values = []
        for representation in REPRESENTATIONS:
            row = metrics[
                (metrics["representation"] == representation) & (metrics["model"] == model)
            ]
            values.append(float(row["balanced_accuracy"].iloc[0]))
        axis.bar(
            x + (model_index - 1) * width,
            values,
            width,
            label=model.replace("_", " ").title(),
            color=color,
        )
    axis.axhline(1.0 / 3.0, color="#9b2c2c", linestyle="--", linewidth=1.2, label="Chance (1/3)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("LOBO balanced accuracy")
    axis.set_title("Independent image-representation comparison")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop_dir", type=Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--sampling_dir", type=Path, default=DEFAULT_SAMPLING_DIR)
    parser.add_argument("--midrib_results", type=Path, default=DEFAULT_MIDRIB_RESULTS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--qc_examples", type=int, default=3)
    parser.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=0.01,
        help="Normalized scikit-image CLAHE clip limit (0-1).",
    )
    parser.add_argument("--clahe_grid_size", nargs=2, type=int, default=(8, 8), metavar=("X", "Y"))
    parser.add_argument(
        "--reuse_features",
        action="store_true",
        help="Reuse an existing representation curve_features.csv, but rerun classification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requested = list(dict.fromkeys(args.representations))
    for representation in requested:
        representation_dir = args.output_root / representation
        feature_csv = representation_dir / "feature_analysis" / "curve_features.csv"
        print(f"\n=== {REPRESENTATION_LABELS[representation]} ===", flush=True)
        if args.reuse_features and feature_csv.exists():
            print(f"Reusing features: {feature_csv}", flush=True)
        else:
            features, _ = build_curve_features(
                representation,
                args.crop_dir,
                args.sampling_dir,
                args.midrib_results,
                feature_csv,
                representation_dir / "qc_examples",
                qc_examples=args.qc_examples,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_grid_size=tuple(args.clahe_grid_size),
            )
            print(f"Wrote feature table {features.shape} to {feature_csv}", flush=True)
        run_classification(
            feature_csv,
            args.labels,
            representation_dir / "treatment_classification_3class",
        )

    if set(requested) == set(REPRESENTATIONS):
        metrics, best = summarize_experiment(args.output_root)
        print("\nBest model per representation")
        print(
            best[["representation_label", "model", "accuracy", "balanced_accuracy", "macro_f1"]]
            .to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
        print(f"\nComparison saved to: {args.output_root / 'comparison'}")

    manifest = {
        "representations_run": requested,
        "features_combined_across_representations": False,
        "fixed_geometry": {
            "sampling_lines": str(args.sampling_dir),
            "midrib_results": str(args.midrib_results),
        },
        "formulas": {name: REPRESENTATION_FORMULAS[name] for name in requested},
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_grid_size": list(args.clahe_grid_size),
        "classification": "three-class LOBO; Low={0,60}, Medium={85,120,153}, High={180,240}",
        "label_sheet": "Tall",
        "plot_treatment_mapping_used": "comparison/plot_treatment_mapping_used.csv",
        "qc_note": "QC PNGs use within-image 1st-99th percentile scaling for visualization only.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
