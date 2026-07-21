"""Find the best two of ten NGRDI half-curves without treating curves as samples.

All 45 fixed two-curve combinations are scanned with the same plot-level
leave-one-block-out workflow. That ranking is exploratory because it uses all
OOF results to rank pairs. A nested leave-one-block-out analysis additionally
selects the curve pair inside each outer training fold and gives the less
optimistic performance estimate of the pair-selection procedure.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from itertools import combinations
import json
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.select_features_classify_treatment as classification


DEFAULT_INPUT = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/image_representation_experiment/"
    "ngrdi/feature_analysis/curve_features.csv"
)
DEFAULT_LABELS = Path(
    r"C:\Users\98748\OneDrive - purdue.edu\PlantSensorLab\2026Summer\n_rate_plot_st_corrected.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/ngrdi_two_curve_pair_experiment"
)

POSITION_COLUMNS = (classification.CURVE_COLUMNS[0], classification.CURVE_COLUMNS[1])
MODELS = ("logistic_l2", "nearest_centroid", "pls_da")
CLASS_ORDER = list(classification.THREE_CLASS_COUNTS)


def position_label(position: tuple[int, str]) -> str:
    """Return a user-facing one-based sampling-line/side label."""
    sample_id, side = position
    return f"line_{int(sample_id) + 1}_{side}"


def identify_curve_positions(data: pd.DataFrame) -> list[tuple[int, str]]:
    """Validate the expected five longitudinal positions by two sides."""
    positions = sorted(
        {
            (int(sample_id), str(side))
            for sample_id, side in data[list(POSITION_COLUMNS)].itertuples(index=False, name=None)
        },
        key=lambda item: (item[0], 0 if item[1] == "upper" else 1, item[1]),
    )
    expected = [(sample_id, side) for sample_id in range(5) for side in ("upper", "lower")]
    if positions != expected:
        raise AssertionError(f"Expected ten curve positions {expected}, found {positions}")
    counts = data.groupby(list(POSITION_COLUMNS)).size()
    if not counts.eq(data[classification.LEAF_COLUMN].nunique()).all():
        raise AssertionError(f"A curve position is missing from one or more leaves:\n{counts}")
    return positions


def aggregate_pair_to_plots(
    data: pd.DataFrame,
    feature_columns: list[str],
    pair: tuple[tuple[int, str], tuple[int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exactly two fixed half-curves per leaf, then leaves to plots."""
    selected = pd.MultiIndex.from_tuples(pair, names=POSITION_COLUMNS)
    row_index = pd.MultiIndex.from_frame(data[list(POSITION_COLUMNS)])
    filtered = data[row_index.isin(selected)].copy()
    leaf_keys = [classification.PLOT_COLUMN, classification.LEAF_COLUMN]
    per_leaf = filtered.groupby(leaf_keys).size()
    if len(per_leaf) != 105 or not per_leaf.eq(2).all():
        raise AssertionError(
            f"Pair {pair} must contribute exactly two curves to each of 105 leaves"
        )

    metadata_columns = [
        classification.TREATMENT_COLUMN,
        classification.BLOCK_COLUMN,
        classification.ORIGINAL_RATE_COLUMN,
    ]
    leaf_features = filtered.groupby(leaf_keys, sort=True)[feature_columns].median()
    leaf_metadata = filtered.groupby(leaf_keys, sort=True)[metadata_columns].first()
    leaf_table = pd.concat([leaf_metadata, leaf_features], axis=1).reset_index()
    plot_features = leaf_table.groupby(classification.PLOT_COLUMN, sort=True)[feature_columns].median()
    plot_metadata = leaf_table.groupby(classification.PLOT_COLUMN, sort=True)[metadata_columns].first()
    plot_table = pd.concat([plot_metadata, plot_features], axis=1).reset_index()
    classification.validate_experimental_design(plot_table, classification.THREE_CLASS_COUNTS)
    return leaf_table, plot_table


def exploratory_pair_scan(
    pair_tables: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every fixed pair with ordinary LOBO, for descriptive ranking."""
    metric_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    selected_tables: list[pd.DataFrame] = []
    expected_test_counts = {label: count // 3 for label, count in classification.THREE_CLASS_COUNTS.items()}
    for pair_index, (pair_id, plot_table) in enumerate(pair_tables.items(), start=1):
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions, metrics, selected, _ = classification.run_leave_one_block_out(
                plot_table,
                feature_columns,
                CLASS_ORDER,
                expected_test_counts,
            )
        first, second = pair_id.split("+")
        for table in (predictions, metrics, selected):
            table.insert(0, "pair_id", pair_id)
            table.insert(1, "curve_1", first)
            table.insert(2, "curve_2", second)
        prediction_tables.append(predictions)
        metric_tables.append(metrics)
        selected_tables.append(selected)
        if pair_index == 1 or pair_index % 10 == 0 or pair_index == len(pair_tables):
            print(f"Exploratory scan: {pair_index}/{len(pair_tables)} pairs", flush=True)

    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics = metrics.sort_values(
        ["model", "balanced_accuracy", "macro_f1", "accuracy", "pair_id"],
        ascending=[True, False, False, False, True],
    )
    metrics["rank_within_model"] = metrics.groupby("model").cumcount() + 1
    predictions = pd.concat(prediction_tables, ignore_index=True)
    selected = pd.concat(selected_tables, ignore_index=True)
    overall = (
        metrics.groupby(["pair_id", "curve_1", "curve_2"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            best_model_accuracy=("accuracy", "max"),
            best_model_balanced_accuracy=("balanced_accuracy", "max"),
        )
        .sort_values(
            ["mean_balanced_accuracy", "mean_macro_f1", "mean_accuracy", "pair_id"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    overall["overall_pair_rank"] = np.arange(1, len(overall) + 1)
    return metrics.reset_index(drop=True), overall, predictions, selected


def nested_pair_selection(
    pair_tables: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a curve pair inside each outer training fold, separately by model."""
    classes = np.asarray(CLASS_ORDER)
    outer_predictions: list[dict[str, object]] = []
    chosen_rows: list[dict[str, object]] = []
    inner_metric_rows: list[dict[str, object]] = []
    outer_selected_tables: list[pd.DataFrame] = []
    blocks = sorted(next(iter(pair_tables.values()))[classification.BLOCK_COLUMN].unique())

    for outer_block in blocks:
        inner_blocks = [block for block in blocks if block != outer_block]
        pair_inner_metrics: dict[str, pd.DataFrame] = {}
        for pair_index, (pair_id, plot_table) in enumerate(pair_tables.items(), start=1):
            inner_prediction_rows: list[dict[str, object]] = []
            for validation_block in inner_blocks:
                inner_train = (
                    (plot_table[classification.BLOCK_COLUMN] != outer_block)
                    & (plot_table[classification.BLOCK_COLUMN] != validation_block)
                )
                inner_test = plot_table[classification.BLOCK_COLUMN] == validation_block
                train_plots = set(plot_table.loc[inner_train, classification.PLOT_COLUMN])
                test_plots = set(plot_table.loc[inner_test, classification.PLOT_COLUMN])
                assert train_plots.isdisjoint(test_plots)
                assert len(train_plots) == 7 and len(test_plots) == 7
                x_train = plot_table.loc[inner_train, feature_columns]
                x_test = plot_table.loc[inner_test, feature_columns]
                y_train = plot_table.loc[inner_train, classification.TREATMENT_COLUMN].to_numpy()
                y_test = plot_table.loc[inner_test, classification.TREATMENT_COLUMN].to_numpy()
                with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    train_selected, test_selected, _, _, _ = classification.select_fold_features(
                        x_train,
                        x_test,
                        y_train,
                        f"outer_{outer_block}_inner_test_{validation_block}_{pair_id}",
                    )
                    predictions = classification.fit_predict_models(
                        train_selected, y_train, test_selected, classes
                    )
                for model, predicted in predictions.items():
                    for true_value, predicted_value in zip(y_test, predicted, strict=True):
                        inner_prediction_rows.append(
                            {
                                "model": model,
                                "true_treatment": true_value,
                                "predicted_treatment": predicted_value,
                            }
                        )
            inner_predictions = pd.DataFrame(inner_prediction_rows)
            inner_metrics = classification.evaluate_predictions(inner_predictions)
            inner_metrics.insert(0, "outer_test_block", outer_block)
            inner_metrics.insert(1, "pair_id", pair_id)
            pair_inner_metrics[pair_id] = inner_metrics
            inner_metric_rows.extend(inner_metrics.to_dict("records"))
            if pair_index % 15 == 0:
                print(
                    f"Nested selection outer block {outer_block}: {pair_index}/{len(pair_tables)} pairs",
                    flush=True,
                )

        all_inner_metrics = pd.concat(pair_inner_metrics.values(), ignore_index=True)
        selected_pair_by_model: dict[str, str] = {}
        for model in MODELS:
            candidates = all_inner_metrics[all_inner_metrics["model"] == model].sort_values(
                ["balanced_accuracy", "macro_f1", "accuracy", "pair_id"],
                ascending=[False, False, False, True],
            )
            winner = candidates.iloc[0]
            selected_pair_by_model[model] = str(winner["pair_id"])
            chosen_rows.append(
                {
                    "outer_test_block": outer_block,
                    "model": model,
                    "selected_pair": winner["pair_id"],
                    "inner_accuracy": winner["accuracy"],
                    "inner_balanced_accuracy": winner["balanced_accuracy"],
                    "inner_macro_f1": winner["macro_f1"],
                }
            )

        outer_cache: dict[str, tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray, pd.DataFrame]] = {}
        for pair_id in sorted(set(selected_pair_by_model.values())):
            plot_table = pair_tables[pair_id]
            train_mask = plot_table[classification.BLOCK_COLUMN] != outer_block
            test_mask = ~train_mask
            train_plots = set(plot_table.loc[train_mask, classification.PLOT_COLUMN])
            test_plots = set(plot_table.loc[test_mask, classification.PLOT_COLUMN])
            assert train_plots.isdisjoint(test_plots)
            x_train = plot_table.loc[train_mask, feature_columns]
            x_test = plot_table.loc[test_mask, feature_columns]
            y_train = plot_table.loc[train_mask, classification.TREATMENT_COLUMN].to_numpy()
            y_test = plot_table.loc[test_mask, classification.TREATMENT_COLUMN].to_numpy()
            with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_selected, test_selected, _, selected, _ = classification.select_fold_features(
                    x_train, x_test, y_train, f"outer_test_{outer_block}_{pair_id}"
                )
                predictions = classification.fit_predict_models(
                    train_selected, y_train, test_selected, classes
                )
            test_metadata = plot_table.loc[
                test_mask,
                [classification.PLOT_COLUMN, classification.BLOCK_COLUMN, classification.ORIGINAL_RATE_COLUMN],
            ].reset_index(drop=True)
            outer_cache[pair_id] = (predictions, test_metadata, y_test, selected)

        for model, pair_id in selected_pair_by_model.items():
            predictions, test_metadata, y_test, selected = outer_cache[pair_id]
            predicted = predictions[model]
            for row_index in range(len(test_metadata)):
                outer_predictions.append(
                    {
                        classification.PLOT_COLUMN: test_metadata.loc[row_index, classification.PLOT_COLUMN],
                        classification.BLOCK_COLUMN: test_metadata.loc[row_index, classification.BLOCK_COLUMN],
                        classification.ORIGINAL_RATE_COLUMN: test_metadata.loc[
                            row_index, classification.ORIGINAL_RATE_COLUMN
                        ],
                        "true_treatment": y_test[row_index],
                        "predicted_treatment": predicted[row_index],
                        "model": model,
                        "selected_pair": pair_id,
                        "correct": bool(predicted[row_index] == y_test[row_index]),
                    }
                )
            selected_copy = selected.copy()
            selected_copy.insert(0, "model", model)
            selected_copy.insert(1, "selected_pair", pair_id)
            selected_copy.insert(2, "outer_test_block", outer_block)
            outer_selected_tables.append(selected_copy)

    predictions = pd.DataFrame(outer_predictions)
    assert predictions.groupby("model")[classification.PLOT_COLUMN].nunique().eq(21).all()
    metrics = classification.evaluate_predictions(predictions)
    chosen = pd.DataFrame(chosen_rows)
    inner_metrics = pd.DataFrame(inner_metric_rows)
    selected_features = pd.concat(outer_selected_tables, ignore_index=True)
    return predictions, metrics, chosen, inner_metrics, selected_features


def save_pair_heatmaps(metrics: pd.DataFrame, positions: list[tuple[int, str]], output_path: Path) -> None:
    labels = [position_label(position).replace("line_", "L").replace("upper", "U").replace("lower", "D") for position in positions]
    label_to_index = {position_label(position): index for index, position in enumerate(positions)}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for axis, model in zip(axes, MODELS, strict=True):
        matrix = np.full((len(positions), len(positions)), np.nan)
        model_metrics = metrics[metrics["model"] == model]
        for row in model_metrics.itertuples(index=False):
            first = label_to_index[row.curve_1]
            second = label_to_index[row.curve_2]
            matrix[first, second] = row.balanced_accuracy
            matrix[second, first] = row.balanced_accuracy
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(0.6, np.nanmax(matrix)))
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_title(model.replace("_", " ").title())
        for row_index in range(len(labels)):
            for column_index in range(len(labels)):
                if np.isfinite(matrix[row_index, column_index]):
                    axis.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if matrix[row_index, column_index] < 0.35 else "black",
                    )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Exploratory NGRDI two-curve pair balanced accuracy\n(pair ranking uses all OOF predictions)")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_top_pairs_plot(overall: pd.DataFrame, output_path: Path) -> None:
    top = overall.head(12).sort_values("mean_balanced_accuracy", ascending=True)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["pair_id"], top["mean_balanced_accuracy"], color="#477f9f")
    axis.axvline(1 / 3, color="#9b2c2c", linestyle="--", label="Chance (1/3)")
    axis.set_xlim(0, max(0.65, float(top["mean_balanced_accuracy"].max()) + 0.05))
    axis.set_xlabel("Mean balanced accuracy across three models")
    axis.set_title("Top exploratory fixed NGRDI curve pairs")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_best_pair_feature_interpretation(
    selected_features: pd.DataFrame,
    best_pair_id: str,
    plot_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save fold-stability and descriptive values for the best fixed pair."""
    selected = selected_features[selected_features["pair_id"] == best_pair_id].copy()
    frequency = (
        selected.groupby("feature", as_index=False)
        .agg(
            folds_selected=("fold", "nunique"),
            mean_rank=("rank", "mean"),
            mean_anova_f=("anova_f", "mean"),
            mean_mrmr_score=("mrmr_score", "mean"),
        )
        .sort_values(["folds_selected", "mean_rank", "feature"], ascending=[False, True, True])
    )
    frequency.to_csv(output_dir / "best_pair_selected_feature_frequency.csv", index=False)
    stable_feature = str(frequency.iloc[0]["feature"])
    values = plot_table[
        [
            classification.PLOT_COLUMN,
            classification.BLOCK_COLUMN,
            classification.ORIGINAL_RATE_COLUMN,
            classification.TREATMENT_COLUMN,
            stable_feature,
        ]
    ].copy()
    values.to_csv(output_dir / "best_pair_stable_feature_plot_values.csv", index=False)
    values.groupby(classification.TREATMENT_COLUMN)[stable_feature].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).to_csv(output_dir / "best_pair_stable_feature_class_summary.csv")

    classes = np.asarray(CLASS_ORDER)
    prediction_rows: list[dict[str, object]] = []
    for test_block in sorted(plot_table[classification.BLOCK_COLUMN].unique()):
        train_mask = plot_table[classification.BLOCK_COLUMN] != test_block
        test_mask = ~train_mask
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            imputer.fit_transform(plot_table.loc[train_mask, [stable_feature]])
        )
        x_test = scaler.transform(
            imputer.transform(plot_table.loc[test_mask, [stable_feature]])
        )
        y_train = plot_table.loc[train_mask, classification.TREATMENT_COLUMN].to_numpy()
        y_test = plot_table.loc[test_mask, classification.TREATMENT_COLUMN].to_numpy()
        predictions = classification.fit_predict_models(x_train, y_train, x_test, classes)
        metadata = plot_table.loc[
            test_mask,
            [classification.PLOT_COLUMN, classification.BLOCK_COLUMN, classification.ORIGINAL_RATE_COLUMN],
        ].reset_index(drop=True)
        for model, predicted in predictions.items():
            for row_index in range(len(metadata)):
                prediction_rows.append(
                    {
                        classification.PLOT_COLUMN: metadata.loc[row_index, classification.PLOT_COLUMN],
                        classification.BLOCK_COLUMN: metadata.loc[row_index, classification.BLOCK_COLUMN],
                        classification.ORIGINAL_RATE_COLUMN: metadata.loc[
                            row_index, classification.ORIGINAL_RATE_COLUMN
                        ],
                        "true_treatment": y_test[row_index],
                        "predicted_treatment": predicted[row_index],
                        "model": model,
                        "correct": bool(predicted[row_index] == y_test[row_index]),
                    }
                )
    feature_only_predictions = pd.DataFrame(prediction_rows)
    feature_only_predictions.to_csv(
        output_dir / "stable_feature_only_oof_predictions.csv", index=False
    )
    classification.evaluate_predictions(feature_only_predictions).to_csv(
        output_dir / "stable_feature_only_oof_metrics.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    markers = {1: "o", 2: "s", 3: "^"}
    colors = {"Low": "#4c78a8", "Medium": "#f2a541", "High": "#c44e52"}
    for block, block_data in values.groupby(classification.BLOCK_COLUMN):
        axes[0].scatter(
            block_data[classification.ORIGINAL_RATE_COLUMN],
            block_data[stable_feature],
            marker=markers[int(block)],
            c=[colors[value] for value in block_data[classification.TREATMENT_COLUMN]],
            edgecolor="black",
            linewidth=0.4,
            s=58,
            label=f"block {block}",
        )
    rate_medians = values.groupby(classification.ORIGINAL_RATE_COLUMN)[stable_feature].median()
    axes[0].plot(rate_medians.index, rate_medians.values, color="#333333", linewidth=1.0, alpha=0.7)
    axes[0].set_xlabel("N rate")
    axes[0].set_ylabel("Feature value")
    axes[0].set_title("Plot-level values by N rate")
    axes[0].grid(color="#dddddd", linewidth=0.6)
    axes[0].legend()

    class_order = ["Low", "Medium", "High"]
    class_values = [
        values.loc[values[classification.TREATMENT_COLUMN] == label, stable_feature].to_numpy()
        for label in class_order
    ]
    axes[1].boxplot(class_values, tick_labels=class_order, showmeans=True)
    for class_index, (label, y_values) in enumerate(zip(class_order, class_values, strict=True), start=1):
        axes[1].scatter(
            np.full(len(y_values), class_index),
            y_values,
            color=colors[label],
            edgecolor="black",
            linewidth=0.35,
            s=38,
            alpha=0.85,
        )
    axes[1].set_xlabel("Three-class treatment")
    axes[1].set_ylabel("Feature value")
    axes[1].set_title("Descriptive class distributions")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.suptitle(
        f"Best NGRDI pair: {best_pair_id}\n{stable_feature}\n"
        "Descriptive all-plot visualization; OOF feature selection remained fold-specific",
        fontsize=11,
    )
    fig.savefig(output_dir / "best_pair_stable_feature_plot.png", dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = classification.read_table(args.input)
    data = classification.attach_design_columns(data, args.labels)
    data[classification.ORIGINAL_RATE_COLUMN] = data[classification.TREATMENT_COLUMN]
    data[classification.TREATMENT_COLUMN] = data[classification.ORIGINAL_RATE_COLUMN].map(
        classification.THREE_CLASS_GROUPS
    )
    if data[classification.TREATMENT_COLUMN].isna().any():
        raise ValueError("One or more N rates are absent from the three-class grouping")
    feature_columns = classification.identify_structure(data)
    positions = identify_curve_positions(data)
    pairs = list(combinations(positions, 2))
    if len(pairs) != 45:
        raise AssertionError(f"Expected 45 two-curve combinations, found {len(pairs)}")

    pair_tables: dict[str, pd.DataFrame] = {}
    leaf_tables: dict[str, pd.DataFrame] = {}
    pair_key_rows: list[dict[str, object]] = []
    for pair in pairs:
        curve_1, curve_2 = map(position_label, pair)
        pair_id = f"{curve_1}+{curve_2}"
        leaf_table, plot_table = aggregate_pair_to_plots(data, feature_columns, pair)
        leaf_tables[pair_id] = leaf_table
        pair_tables[pair_id] = plot_table
        pair_key_rows.append(
            {
                "pair_id": pair_id,
                "curve_1": curve_1,
                "curve_2": curve_2,
                "curve_1_sample_id": pair[0][0],
                "curve_1_side": pair[0][1],
                "curve_2_sample_id": pair[1][0],
                "curve_2_side": pair[1][1],
            }
        )
    print("Prepared 45 fixed NGRDI two-curve plot tables", flush=True)

    metrics, overall, predictions, exploratory_selected = exploratory_pair_scan(
        pair_tables, feature_columns
    )
    nested_predictions, nested_metrics, chosen, inner_metrics, nested_selected = nested_pair_selection(
        pair_tables, feature_columns
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_key_rows).to_csv(args.output_dir / "curve_pair_key.csv", index=False)
    metrics.to_csv(args.output_dir / "exploratory_pair_model_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "exploratory_pair_overall_ranking.csv", index=False)
    predictions.to_csv(args.output_dir / "exploratory_pair_oof_predictions.csv", index=False)
    exploratory_selected.to_csv(args.output_dir / "exploratory_selected_features.csv", index=False)
    nested_predictions.to_csv(args.output_dir / "nested_oof_predictions.csv", index=False)
    nested_metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    chosen.to_csv(args.output_dir / "nested_selected_pairs_by_outer_fold.csv", index=False)
    inner_metrics.to_csv(args.output_dir / "nested_inner_pair_metrics.csv", index=False)
    nested_selected.to_csv(args.output_dir / "nested_outer_selected_features.csv", index=False)
    save_pair_heatmaps(metrics, positions, args.output_dir / "exploratory_pair_balanced_accuracy_heatmaps.png")
    save_top_pairs_plot(overall, args.output_dir / "exploratory_top_pairs.png")
    classification.save_confusion_outputs(nested_predictions, CLASS_ORDER, args.output_dir)

    all_ten_metrics_path = args.input.parents[1] / "treatment_classification_3class" / "oof_metrics.csv"
    if all_ten_metrics_path.exists():
        all_ten_metrics = pd.read_csv(all_ten_metrics_path)
        all_ten_metrics.to_csv(args.output_dir / "reference_all_10_curves_metrics.csv", index=False)

    best_by_model = metrics[metrics["rank_within_model"] == 1].copy()
    best_by_model.to_csv(args.output_dir / "exploratory_best_pair_by_model.csv", index=False)
    best_pair_id = str(overall.iloc[0]["pair_id"])
    save_best_pair_feature_interpretation(
        exploratory_selected,
        best_pair_id,
        pair_tables[best_pair_id],
        args.output_dir,
    )
    best_prediction_tables = []
    for row in best_by_model.itertuples(index=False):
        best_prediction_tables.append(
            predictions[
                (predictions["model"] == row.model) & (predictions["pair_id"] == row.pair_id)
            ]
        )
    best_fixed_predictions = pd.concat(best_prediction_tables, ignore_index=True)
    best_fixed_predictions.to_csv(
        args.output_dir / "exploratory_best_pair_oof_predictions.csv", index=False
    )
    best_fixed_dir = args.output_dir / "exploratory_best_pair_confusion"
    best_fixed_dir.mkdir(parents=True, exist_ok=True)
    classification.save_confusion_outputs(best_fixed_predictions, CLASS_ORDER, best_fixed_dir)
    comparison_rows: list[dict[str, object]] = []
    if all_ten_metrics_path.exists():
        for method, table, pair_column, note in (
            ("all_10_curves", all_ten_metrics, None, "Pre-specified baseline"),
            (
                "best_fixed_2_curves_exploratory",
                best_by_model,
                "pair_id",
                "Optimistic: best pair chosen from 45 using all OOF results",
            ),
            (
                "nested_select_2_curves",
                nested_metrics,
                None,
                "Pair chosen inside each outer training fold",
            ),
        ):
            for row in table.itertuples(index=False):
                comparison_rows.append(
                    {
                        "method": method,
                        "model": row.model,
                        "selected_pair": getattr(row, pair_column) if pair_column else "varies/not_applicable",
                        "accuracy": row.accuracy,
                        "balanced_accuracy": row.balanced_accuracy,
                        "macro_f1": row.macro_f1,
                        "interpretation": note,
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(
            args.output_dir / "comparison_all10_vs_two_curve.csv", index=False
        )
    manifest = {
        "image_representation": "NGRDI only",
        "candidate_curve_positions": [position_label(position) for position in positions],
        "candidate_pairs": len(pairs),
        "independent_unit": "plot",
        "outer_validation": "leave-one-block-out",
        "exploratory_warning": "The fixed-pair ranking uses all 21 OOF predictions and is optimistic after choosing among 45 pairs.",
        "nested_interpretation": "Nested OOF metrics select the pair inside each outer training fold and estimate the selection procedure.",
        "features_combined_with_other_representations": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nExploratory best fixed pair by model")
    print(
        best_by_model[["model", "pair_id", "accuracy", "balanced_accuracy", "macro_f1"]]
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nNested OOF metrics (pair selected inside outer training folds)")
    print(nested_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
