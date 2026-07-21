"""Leakage-safe feature selection and seven-class treatment classification.

The independent experimental unit is the plot. Curves are first aggregated
to leaves and leaves to plots. Every learned preprocessing step is then fit
inside a leave-one-block-out training fold only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration: edit these names if a future workbook uses different fields.
# ---------------------------------------------------------------------------
INPUT_FILE = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/feature_analysis/"
    "leaf2_green_curve_features.csv"
)
INPUT_SHEET: str | int = 0

# Used only when treatment is not already present in INPUT_FILE.
LABEL_FILE: Path | None = Path(
    r"C:\Users\98748\OneDrive - purdue.edu\PlantSensorLab\2026Summer\n_rate_plot_st_corrected.xlsx"
)
LABEL_SHEET: str | int = "Tall"
LABEL_HEADER_ROW = 0
LABEL_PLOT_COLUMN = "Plot"
LABEL_TREATMENT_COLUMN = "Treatment"

TREATMENT_COLUMN = "treatment"
ORIGINAL_RATE_COLUMN = "n_rate"
BLOCK_COLUMN = "block"
PLOT_COLUMN = "plot_number"
LEAF_COLUMN = "source_profile_file"
CURVE_COLUMNS = ("sample_id", "midrib_side")
PLOT_TO_BLOCK_DIVISOR = 100

EXCLUDE_FROM_FEATURES = (
    "prefix",
    "geno",
    "plant_number",
    "leaf",
    "timestamp",
    "filename",
    "source_profile_file",
    "sample_id",
    "midrib_side",
    "n_rate",
)

EXPECTED_BLOCKS = 3
EXPECTED_LEAVES_PER_PLOT = 5
EXPECTED_CURVES_PER_LEAF = 10

SEVEN_CLASS_COUNTS = {0: 3, 60: 3, 85: 3, 120: 3, 153: 3, 180: 3, 240: 3}
THREE_CLASS_GROUPS = {
    0: "Low",
    60: "Low",
    85: "Medium",
    120: "Medium",
    153: "Medium",
    180: "High",
    240: "High",
}
THREE_CLASS_COUNTS = {"Low": 6, "Medium": 9, "High": 6}

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-12
CORRELATION_THRESHOLD = 0.95
ANOVA_TOP_K = 20
MRMR_TOP_K = 3
LOGISTIC_C = 1.0
PLS_MAX_COMPONENTS = 2

DEFAULT_OUTPUT_DIR = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/treatment_classification"
)
THREE_CLASS_OUTPUT_DIR = Path(
    "outputs/RGB_tall_v9_leaf2_centerline_sampling/treatment_classification_3class"
)


def read_table(path: Path, *, sheet: str | int = 0, header: int = 0) -> pd.DataFrame:
    """Read CSV/TSV or Excel while preserving column names."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", header=header)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name=sheet, header=header)
        except ImportError as exc:
            raise ImportError("Excel input requires openpyxl. Run: pip install -r requirements.txt") from exc
    raise ValueError(f"Unsupported input format: {path}")


def attach_design_columns(data: pd.DataFrame, label_file: Path | None) -> pd.DataFrame:
    """Attach treatment labels and derive block when they are absent."""
    output = data.copy()
    if TREATMENT_COLUMN not in output.columns:
        if label_file is None:
            raise ValueError(
                f"{TREATMENT_COLUMN!r} is absent and no label file was configured."
            )
        labels = read_table(label_file, sheet=LABEL_SHEET, header=LABEL_HEADER_ROW)
        required = {LABEL_PLOT_COLUMN, LABEL_TREATMENT_COLUMN}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"Missing label columns: {sorted(missing)}")
        labels = labels[[LABEL_PLOT_COLUMN, LABEL_TREATMENT_COLUMN]].copy()
        label_conflicts = labels.groupby(LABEL_PLOT_COLUMN)[LABEL_TREATMENT_COLUMN].nunique(dropna=False)
        label_conflicts = label_conflicts[label_conflicts > 1]
        if not label_conflicts.empty:
            raise ValueError(
                "Treatment is inconsistent within label-sheet plots: "
                f"{label_conflicts.index.tolist()}"
            )
        labels = labels.rename(
            columns={LABEL_PLOT_COLUMN: PLOT_COLUMN, LABEL_TREATMENT_COLUMN: TREATMENT_COLUMN}
        )
        labels = labels.drop_duplicates(PLOT_COLUMN)
        output = output.merge(labels, how="left", on=PLOT_COLUMN, validate="many_to_one")

    if BLOCK_COLUMN not in output.columns:
        numeric_plot = pd.to_numeric(output[PLOT_COLUMN], errors="coerce")
        if numeric_plot.isna().any():
            raise ValueError(f"Cannot derive {BLOCK_COLUMN!r} from nonnumeric plot IDs")
        output[BLOCK_COLUMN] = (numeric_plot // PLOT_TO_BLOCK_DIVISOR).astype(int)
    return output


def identify_structure(data: pd.DataFrame) -> list[str]:
    """Validate configured metadata and identify numeric feature columns."""
    required = {
        TREATMENT_COLUMN,
        BLOCK_COLUMN,
        PLOT_COLUMN,
        LEAF_COLUMN,
        *CURVE_COLUMNS,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Configured columns missing from input: {sorted(missing)}")

    excluded = {
        TREATMENT_COLUMN,
        BLOCK_COLUMN,
        PLOT_COLUMN,
        LEAF_COLUMN,
        *CURVE_COLUMNS,
        *EXCLUDE_FROM_FEATURES,
    }
    feature_columns = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns remain after metadata exclusion")

    print("\nIdentified data structure")
    print(f"  treatment column : {TREATMENT_COLUMN}")
    print(f"  block column     : {BLOCK_COLUMN}")
    print(f"  plot ID          : {PLOT_COLUMN}")
    print(f"  leaf ID          : {LEAF_COLUMN}")
    print(f"  curve ID         : {list(CURVE_COLUMNS)}")
    print(f"  numeric features : {len(feature_columns)}")
    print(f"  input dimensions : {data.shape}")
    return feature_columns


def aggregate_to_plots(
    data: pd.DataFrame,
    feature_columns: list[str],
    expected_class_counts: dict[Any, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-aggregate 10 curves to leaves and five leaves to plots."""
    missing_treatment = data.loc[data[TREATMENT_COLUMN].isna(), PLOT_COLUMN].drop_duplicates().tolist()
    if missing_treatment:
        raise ValueError(
            "Treatment is missing for plot(s): "
            f"{missing_treatment}. Correct the input/label mapping before classification."
        )

    metadata_columns = [TREATMENT_COLUMN, BLOCK_COLUMN]
    if ORIGINAL_RATE_COLUMN in data.columns:
        metadata_columns.append(ORIGINAL_RATE_COLUMN)
    plot_consistency = data.groupby(PLOT_COLUMN)[metadata_columns].nunique(dropna=False)
    inconsistent = plot_consistency[(plot_consistency > 1).any(axis=1)]
    if not inconsistent.empty:
        raise AssertionError(f"Treatment/block is inconsistent within plots:\n{inconsistent}")

    curve_counts = (
        data[[PLOT_COLUMN, LEAF_COLUMN, *CURVE_COLUMNS]]
        .drop_duplicates()
        .groupby([PLOT_COLUMN, LEAF_COLUMN])
        .size()
    )
    bad_curve_counts = curve_counts[curve_counts != EXPECTED_CURVES_PER_LEAF]
    if not bad_curve_counts.empty:
        raise AssertionError(
            f"Expected {EXPECTED_CURVES_PER_LEAF} curves per leaf; mismatches:\n"
            f"{bad_curve_counts.to_string()}"
        )

    leaf_counts = data[[PLOT_COLUMN, LEAF_COLUMN]].drop_duplicates().groupby(PLOT_COLUMN).size()
    bad_leaf_counts = leaf_counts[leaf_counts != EXPECTED_LEAVES_PER_PLOT]
    if not bad_leaf_counts.empty:
        raise AssertionError(
            f"Expected {EXPECTED_LEAVES_PER_PLOT} leaves per plot; mismatches:\n"
            f"{bad_leaf_counts.to_string()}"
        )

    leaf_keys = [PLOT_COLUMN, LEAF_COLUMN]
    leaf_features = data.groupby(leaf_keys, sort=True)[feature_columns].median()
    leaf_metadata = data.groupby(leaf_keys, sort=True)[metadata_columns].first()
    leaf_table = pd.concat([leaf_metadata, leaf_features], axis=1).reset_index()
    print(f"After curve -> leaf median : {leaf_table.shape}")

    plot_features = leaf_table.groupby(PLOT_COLUMN, sort=True)[feature_columns].median()
    plot_metadata = leaf_table.groupby(PLOT_COLUMN, sort=True)[metadata_columns].first()
    plot_table = pd.concat([plot_metadata, plot_features], axis=1).reset_index()
    print(f"After leaf -> plot median  : {plot_table.shape}")
    validate_experimental_design(plot_table, expected_class_counts)
    return leaf_table, plot_table


def validate_experimental_design(
    plot_table: pd.DataFrame,
    expected_class_counts: dict[Any, int],
) -> None:
    """Assert the requested treatment grouping and three-block balance."""
    n_plots = plot_table[PLOT_COLUMN].nunique()
    blocks = sorted(plot_table[BLOCK_COLUMN].unique().tolist())
    classes = plot_table[TREATMENT_COLUMN].unique().tolist()
    if len(blocks) != EXPECTED_BLOCKS:
        raise AssertionError(f"Expected {EXPECTED_BLOCKS} blocks, found {blocks}")
    if set(classes) != set(expected_class_counts):
        raise AssertionError(
            f"Expected classes {list(expected_class_counts)}, found {sorted(classes)}"
        )
    if n_plots != sum(expected_class_counts.values()):
        raise AssertionError(f"Expected {sum(expected_class_counts.values())} plots, found {n_plots}")

    class_counts = plot_table.groupby(TREATMENT_COLUMN)[PLOT_COLUMN].nunique().to_dict()
    if class_counts != expected_class_counts:
        raise AssertionError(f"Unexpected plots per class: {class_counts}")
    expected_per_block = {
        label: count // EXPECTED_BLOCKS for label, count in expected_class_counts.items()
    }
    if any(count % EXPECTED_BLOCKS for count in expected_class_counts.values()):
        raise AssertionError("Expected class counts must divide evenly across blocks")
    block_class = pd.crosstab(plot_table[BLOCK_COLUMN], plot_table[TREATMENT_COLUMN]).reindex(
        index=blocks, columns=list(expected_class_counts), fill_value=0
    )
    expected_matrix = pd.DataFrame(
        [expected_per_block] * EXPECTED_BLOCKS,
        index=blocks,
        columns=list(expected_class_counts),
    )
    if not block_class.equals(expected_matrix):
        raise AssertionError(
            "Each block must contain the expected plots per class:\n"
            f"{block_class.to_string()}"
        )


def correlation_representatives(
    x_train: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Complete-linkage cluster by 1-|r| and retain each cluster medoid."""
    if len(feature_names) == 1:
        mapping = pd.DataFrame(
            {"feature": feature_names, "correlation_cluster": [1], "representative": feature_names}
        )
        return x_train, feature_names, mapping

    correlation = np.corrcoef(x_train, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.clip(1.0 - np.abs(correlation), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    cluster_ids = fcluster(tree, t=1.0 - threshold, criterion="distance")
    variances = np.var(x_train, axis=0, ddof=0)

    retained_indices: list[int] = []
    representatives: dict[int, str] = {}
    for cluster_id in sorted(np.unique(cluster_ids)):
        indices = np.flatnonzero(cluster_ids == cluster_id)
        if indices.size == 1:
            chosen = int(indices[0])
        else:
            cluster_corr = np.abs(correlation[np.ix_(indices, indices)])
            centrality = (cluster_corr.sum(axis=1) - 1.0) / (indices.size - 1)
            candidates = sorted(
                indices,
                key=lambda index: (
                    -float(centrality[np.where(indices == index)[0][0]]),
                    -float(variances[index]),
                    feature_names[index],
                ),
            )
            chosen = int(candidates[0])
        retained_indices.append(chosen)
        representatives[int(cluster_id)] = feature_names[chosen]

    retained_indices.sort(key=lambda index: feature_names[index])
    retained_names = [feature_names[index] for index in retained_indices]
    mapping = pd.DataFrame(
        {
            "feature": feature_names,
            "correlation_cluster": cluster_ids,
            "representative": [representatives[int(cluster)] for cluster in cluster_ids],
        }
    )
    return x_train[:, retained_indices], retained_names, mapping


def greedy_mrmr(
    x_train: np.ndarray,
    feature_names: list[str],
    anova_scores: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Select features by normalized ANOVA relevance minus mean |r|."""
    finite_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = float(np.max(finite_scores)) if finite_scores.size else 0.0
    relevance = finite_scores / maximum if maximum > 0 else np.zeros_like(finite_scores)
    correlation = np.corrcoef(x_train, rowvar=False)
    if np.ndim(correlation) == 0:
        correlation = np.array([[1.0]])
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0)

    remaining = set(range(len(feature_names)))
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    while remaining and len(selected) < min(top_k, len(feature_names)):
        candidates: list[tuple[float, float, str, int, float]] = []
        for index in remaining:
            redundancy = float(np.mean(correlation[index, selected])) if selected else 0.0
            score = float(relevance[index] - redundancy)
            candidates.append((score, float(relevance[index]), feature_names[index], index, redundancy))
        score, scaled_relevance, _, chosen, redundancy = sorted(
            candidates, key=lambda item: (-item[0], -item[1], item[2])
        )[0]
        selected.append(chosen)
        remaining.remove(chosen)
        records.append(
            {
                "rank": len(selected),
                "feature": feature_names[chosen],
                "anova_f": float(finite_scores[chosen]),
                "anova_relevance_scaled": scaled_relevance,
                "mean_abs_correlation_to_selected": redundancy,
                "mrmr_score": score,
            }
        )
    return pd.DataFrame(records)


def select_fold_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    fold_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    """Fit every selection/preprocessing operation on the training fold."""
    dimensions: dict[str, Any] = {"fold": fold_name, "initial_features": x_train.shape[1]}

    nonempty = x_train.columns[~x_train.isna().all(axis=0)].tolist()
    x_train = x_train[nonempty]
    x_test = x_test[nonempty]
    dimensions["after_all_missing"] = len(nonempty)
    print(f"  after all-missing removal : {x_train.shape}")

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(x_train)
    test_imputed = imputer.transform(x_test)
    dimensions["imputer_fit_rows"] = x_train.shape[0]
    print(f"  after train-only impute   : {train_imputed.shape}")

    variance = np.var(train_imputed, axis=0, ddof=0)
    variable_mask = np.isfinite(variance) & (variance > NEAR_ZERO_VARIANCE_THRESHOLD)
    variable_names = [name for name, keep in zip(nonempty, variable_mask, strict=True) if keep]
    train_variable = train_imputed[:, variable_mask]
    test_variable = test_imputed[:, variable_mask]
    if not variable_names:
        raise ValueError(f"{fold_name}: no features survived variance filtering")
    dimensions["after_variance"] = len(variable_names)
    print(f"  after constant/NZV       : {train_variable.shape}")

    train_corr, corr_names, corr_mapping = correlation_representatives(
        train_variable, variable_names, CORRELATION_THRESHOLD
    )
    name_to_index = {name: index for index, name in enumerate(variable_names)}
    corr_indices = [name_to_index[name] for name in corr_names]
    test_corr = test_variable[:, corr_indices]
    dimensions["after_correlation"] = len(corr_names)
    print(f"  after |r| clustering     : {train_corr.shape}")

    anova_scores, _ = f_classif(train_corr, y_train)
    ranking = pd.DataFrame({"feature": corr_names, "anova_f": anova_scores})
    ranking["anova_f"] = ranking["anova_f"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ranking = ranking.sort_values(["anova_f", "feature"], ascending=[False, True]).head(ANOVA_TOP_K)
    anova_names = ranking["feature"].tolist()
    corr_name_to_index = {name: index for index, name in enumerate(corr_names)}
    anova_indices = [corr_name_to_index[name] for name in anova_names]
    train_anova = train_corr[:, anova_indices]
    test_anova = test_corr[:, anova_indices]
    anova_f = ranking["anova_f"].to_numpy(dtype=float)
    dimensions["after_anova"] = len(anova_names)
    print(f"  after ANOVA Top {ANOVA_TOP_K:<2}      : {train_anova.shape}")

    selected = greedy_mrmr(train_anova, anova_names, anova_f, MRMR_TOP_K)
    selected_names = selected["feature"].tolist()
    anova_name_to_index = {name: index for index, name in enumerate(anova_names)}
    selected_indices = [anova_name_to_index[name] for name in selected_names]
    train_selected = train_anova[:, selected_indices]
    test_selected = test_anova[:, selected_indices]
    dimensions["after_mrmr"] = len(selected_names)
    print(f"  after greedy mRMR Top {MRMR_TOP_K}: {train_selected.shape}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_selected)
    test_scaled = scaler.transform(test_selected)
    dimensions["scaler_fit_rows"] = train_selected.shape[0]
    selected["fold"] = fold_name
    selected = selected.merge(
        corr_mapping[["feature", "correlation_cluster"]], on="feature", how="left", validate="one_to_one"
    )
    return train_scaled, test_scaled, selected_names, selected, dimensions


def fit_predict_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit the three requested simple classifiers."""
    logistic = LogisticRegression(
        penalty="l2",
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=5000,
    )
    nearest = NearestCentroid()
    logistic.fit(x_train, y_train)
    nearest.fit(x_train, y_train)

    y_one_hot = (y_train[:, None] == classes[None, :]).astype(float)
    n_components = min(PLS_MAX_COMPONENTS, x_train.shape[1], x_train.shape[0] - 1)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x_train, y_one_hot)
    pls_scores = np.asarray(pls.predict(x_test))
    if pls_scores.ndim == 1:
        pls_scores = pls_scores[:, None]
    pls_prediction = classes[np.argmax(pls_scores, axis=1)]
    return {
        "logistic_l2": logistic.predict(x_test),
        "nearest_centroid": nearest.predict(x_test),
        "pls_da": pls_prediction,
    }


def run_leave_one_block_out(
    plot_table: pd.DataFrame,
    feature_columns: list[str],
    class_order: list[Any],
    expected_test_counts: dict[Any, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate leakage-safe out-of-fold predictions for all plots."""
    classes = np.asarray(class_order)
    prediction_rows: list[dict[str, Any]] = []
    selected_tables: list[pd.DataFrame] = []
    dimension_rows: list[dict[str, Any]] = []
    blocks = sorted(plot_table[BLOCK_COLUMN].unique().tolist())

    for test_block in blocks:
        train_mask = plot_table[BLOCK_COLUMN] != test_block
        test_mask = ~train_mask
        train_plots = set(plot_table.loc[train_mask, PLOT_COLUMN])
        test_plots = set(plot_table.loc[test_mask, PLOT_COLUMN])
        assert train_plots.isdisjoint(test_plots), "Plot leakage between training and test data"
        assert len(train_plots) == 14 and len(test_plots) == 7
        test_class_counts = plot_table.loc[test_mask, TREATMENT_COLUMN].value_counts().to_dict()
        assert test_class_counts == expected_test_counts

        fold_name = f"test_block_{test_block}"
        print(f"\n{fold_name}: train plots={len(train_plots)}, test plots={len(test_plots)}")
        x_train = plot_table.loc[train_mask, feature_columns]
        x_test = plot_table.loc[test_mask, feature_columns]
        y_train = plot_table.loc[train_mask, TREATMENT_COLUMN].to_numpy()
        y_test = plot_table.loc[test_mask, TREATMENT_COLUMN].to_numpy()
        train_selected, test_selected, _, selected, dimensions = select_fold_features(
            x_train, x_test, y_train, fold_name
        )
        selected["test_block"] = test_block
        selected_tables.append(selected)
        dimension_rows.append(dimensions)

        predictions = fit_predict_models(train_selected, y_train, test_selected, classes)
        test_metadata_columns = [PLOT_COLUMN, BLOCK_COLUMN]
        if ORIGINAL_RATE_COLUMN in plot_table.columns:
            test_metadata_columns.append(ORIGINAL_RATE_COLUMN)
        test_metadata = plot_table.loc[test_mask, test_metadata_columns].reset_index(drop=True)
        for model_name, predicted in predictions.items():
            for row_index in range(len(test_metadata)):
                prediction_row = {
                    PLOT_COLUMN: test_metadata.loc[row_index, PLOT_COLUMN],
                    BLOCK_COLUMN: test_metadata.loc[row_index, BLOCK_COLUMN],
                    "true_treatment": y_test[row_index],
                    "predicted_treatment": predicted[row_index],
                    "model": model_name,
                    "correct": bool(predicted[row_index] == y_test[row_index]),
                }
                if ORIGINAL_RATE_COLUMN in test_metadata.columns:
                    prediction_row[ORIGINAL_RATE_COLUMN] = test_metadata.loc[
                        row_index, ORIGINAL_RATE_COLUMN
                    ]
                prediction_rows.append(prediction_row)

    predictions = pd.DataFrame(prediction_rows)
    assert predictions.groupby("model")[PLOT_COLUMN].nunique().eq(len(plot_table)).all()
    selected_features = pd.concat(selected_tables, ignore_index=True)
    dimensions = pd.DataFrame(dimension_rows)
    metrics = evaluate_predictions(predictions)
    return predictions, metrics, selected_features, dimensions


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate requested multiclass metrics from out-of-fold predictions."""
    rows = []
    for model_name, model_predictions in predictions.groupby("model", sort=True):
        true = model_predictions["true_treatment"]
        predicted = model_predictions["predicted_treatment"]
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(true, predicted),
                "balanced_accuracy": balanced_accuracy_score(true, predicted),
                "macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def selection_frequency(selected_features: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Summarize how often each mRMR feature was selected."""
    frequency = (
        selected_features.groupby("feature", as_index=False)
        .agg(
            folds_selected=("fold", "nunique"),
            mean_rank=("rank", "mean"),
            mean_anova_f=("anova_f", "mean"),
            mean_mrmr_score=("mrmr_score", "mean"),
        )
    )
    frequency["selection_frequency"] = frequency["folds_selected"] / n_folds
    return frequency.sort_values(
        ["folds_selected", "mean_rank", "feature"], ascending=[False, True, True]
    ).reset_index(drop=True)


def save_confusion_outputs(
    predictions: pd.DataFrame,
    classes: list[Any],
    output_dir: Path,
) -> None:
    """Save confusion matrices as CSV and a three-panel figure."""
    models = sorted(predictions["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.5), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model_name in zip(axes, models, strict=True):
        model_predictions = predictions[predictions["model"] == model_name]
        matrix = confusion_matrix(
            model_predictions["true_treatment"],
            model_predictions["predicted_treatment"],
            labels=classes,
        )
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
            output_dir / f"confusion_matrix_{model_name}.csv", index_label="true_treatment"
        )
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
        axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel("Predicted treatment")
        axis.set_ylabel("True treatment")
        axis.set_title(model_name.replace("_", " ").title())
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Leave-one-block-out out-of-fold confusion matrices")
    fig.savefig(output_dir / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def save_frequency_plot(frequency: pd.DataFrame, output_dir: Path) -> None:
    """Plot fold selection frequency for every selected feature."""
    plot_data = frequency.sort_values(["folds_selected", "mean_rank"], ascending=[True, False])
    height = max(4.0, 0.42 * len(plot_data))
    fig, axis = plt.subplots(figsize=(10, height))
    axis.barh(plot_data["feature"], plot_data["selection_frequency"], color="#3b82a0")
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Selection frequency across three training folds")
    axis.set_ylabel("Feature")
    axis.set_title("Greedy mRMR feature-selection stability")
    axis.grid(axis="x", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_selection_frequency.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(
    plot_table: pd.DataFrame,
    final_features: list[str],
    output_dir: Path,
) -> None:
    """Plot descriptive Pearson correlations for the fold-stable features."""
    correlation = plot_table[final_features].corr(method="pearson")
    short_labels = [f"F{index + 1}" for index in range(len(final_features))]
    fig, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(final_features)), short_labels)
    axis.set_yticks(range(len(final_features)), short_labels)
    for row in range(len(final_features)):
        for column in range(len(final_features)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center")
    axis.set_title(
        "Pearson correlation of top selection-frequency features\n"
        "(descriptive; not used for OOF selection)"
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    feature_key = "\n".join(
        f"{label}: {feature}" for label, feature in zip(short_labels, final_features, strict=True)
    )
    fig.text(0.08, 0.04, feature_key, ha="left", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.82, bottom=0.30)
    fig.savefig(output_dir / "selected_feature_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def save_exploratory_pls_plot(
    plot_table: pd.DataFrame,
    final_features: list[str],
    classes: list[Any],
    output_dir: Path,
) -> None:
    """Fit an all-data PLS-DA for visualization only, never for OOF metrics."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(plot_table[final_features]))
    y = plot_table[TREATMENT_COLUMN].to_numpy()
    y_one_hot = (y[:, None] == np.asarray(classes)[None, :]).astype(float)
    n_components = min(PLS_MAX_COMPONENTS, x.shape[1], x.shape[0] - 1)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(x, y_one_hot)
    scores = np.asarray(pls.x_scores_)
    score_1 = scores[:, 0]
    score_2 = scores[:, 1] if scores.shape[1] >= 2 else np.zeros(len(scores))

    score_table = plot_table[[PLOT_COLUMN, BLOCK_COLUMN, TREATMENT_COLUMN]].copy()
    score_table["PLS1"] = score_1
    score_table["PLS2"] = score_2
    score_table.to_csv(output_dir / "exploratory_pls_da_scores.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 6))
    color_map = plt.get_cmap("tab10")
    class_to_color = {label: color_map(index) for index, label in enumerate(classes)}
    markers = ["o", "s", "^"]
    blocks = sorted(plot_table[BLOCK_COLUMN].unique())
    for block_index, block in enumerate(blocks):
        for treatment in classes:
            mask = (plot_table[BLOCK_COLUMN] == block) & (plot_table[TREATMENT_COLUMN] == treatment)
            axis.scatter(
                score_1[mask],
                score_2[mask],
                color=class_to_color[treatment],
                marker=markers[block_index % len(markers)],
                s=65,
                edgecolor="black",
                linewidth=0.4,
            )
    for index, plot_id in enumerate(plot_table[PLOT_COLUMN]):
        axis.annotate(str(plot_id), (score_1[index], score_2[index]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    treatment_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=class_to_color[label], label=str(label))
        for label in classes
    ]
    block_handles = [
        plt.Line2D([0], [0], marker=markers[index], linestyle="", color="black", label=f"block {block}")
        for index, block in enumerate(blocks)
    ]
    first_legend = axis.legend(handles=treatment_handles, title="Treatment", loc="best")
    axis.add_artist(first_legend)
    axis.legend(handles=block_handles, title="Block", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.axhline(0, color="#cccccc", linewidth=0.8)
    axis.axvline(0, color="#cccccc", linewidth=0.8)
    axis.set_xlabel("PLS score 1")
    axis.set_ylabel("PLS score 2")
    axis.set_title("Exploratory PLS-DA score plot\n(all plots used; not a validation result)")
    fig.tight_layout()
    fig.savefig(output_dir / "exploratory_pls_da_score_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    leaf_table: pd.DataFrame,
    plot_table: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_features: pd.DataFrame,
    dimensions: pd.DataFrame,
    feature_columns: list[str],
    class_order: list[Any],
    treatment_groups: dict[Any, str] | None,
) -> None:
    """Save auditable CSV results and requested figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leaf_table.to_csv(output_dir / "leaf_level_median_features.csv", index=False)
    plot_table.to_csv(output_dir / "plot_level_median_features.csv", index=False)
    predictions.to_csv(output_dir / "oof_plot_predictions.csv", index=False)
    metrics.to_csv(output_dir / "oof_metrics.csv", index=False)
    selected_features.to_csv(output_dir / "selected_features_by_fold.csv", index=False)
    dimensions.to_csv(output_dir / "fold_feature_dimensions.csv", index=False)

    n_folds = plot_table[BLOCK_COLUMN].nunique()
    frequency = selection_frequency(selected_features, n_folds)
    frequency.to_csv(output_dir / "feature_selection_frequency.csv", index=False)
    final_features = frequency.head(MRMR_TOP_K)["feature"].tolist()
    classes = class_order

    save_confusion_outputs(predictions, classes, output_dir)
    save_frequency_plot(frequency, output_dir)
    save_correlation_heatmap(plot_table, final_features, output_dir)
    save_exploratory_pls_plot(plot_table, final_features, classes, output_dir)

    config = {
        "treatment_column": TREATMENT_COLUMN,
        "block_column": BLOCK_COLUMN,
        "plot_column": PLOT_COLUMN,
        "leaf_column": LEAF_COLUMN,
        "curve_columns": list(CURVE_COLUMNS),
        "numeric_feature_count": len(feature_columns),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "anova_top_k": ANOVA_TOP_K,
        "mrmr_top_k": MRMR_TOP_K,
        "final_descriptive_features": final_features,
        "class_order": class_order,
        "treatment_groups": treatment_groups,
        "note": "All OOF preprocessing/selection was fit on training blocks only. The final PLS plot is exploratory.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--sheet", default=INPUT_SHEET)
    parser.add_argument("--labels", type=Path, default=LABEL_FILE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--three_class",
        action="store_true",
        help="Group N rates as Low={0,60}, Medium={85,120,153}, High={180,240}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = read_table(args.input, sheet=args.sheet)
    data = attach_design_columns(data, args.labels)
    if args.three_class:
        data[ORIGINAL_RATE_COLUMN] = data[TREATMENT_COLUMN]
        data[TREATMENT_COLUMN] = data[ORIGINAL_RATE_COLUMN].map(THREE_CLASS_GROUPS)
        unmapped = sorted(data.loc[data[TREATMENT_COLUMN].isna(), ORIGINAL_RATE_COLUMN].dropna().unique())
        if unmapped:
            raise ValueError(f"N rates missing from THREE_CLASS_GROUPS: {unmapped}")
        expected_class_counts = THREE_CLASS_COUNTS
        class_order = list(THREE_CLASS_COUNTS)
        treatment_groups: dict[Any, str] | None = THREE_CLASS_GROUPS
        output_dir = args.output_dir or THREE_CLASS_OUTPUT_DIR
        print(f"Three-class grouping: {THREE_CLASS_GROUPS}")
    else:
        expected_class_counts = SEVEN_CLASS_COUNTS
        class_order = list(SEVEN_CLASS_COUNTS)
        treatment_groups = None
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    feature_columns = identify_structure(data)
    leaf_table, plot_table = aggregate_to_plots(data, feature_columns, expected_class_counts)
    expected_test_counts = {
        label: count // EXPECTED_BLOCKS for label, count in expected_class_counts.items()
    }
    predictions, metrics, selected_features, dimensions = run_leave_one_block_out(
        plot_table,
        feature_columns,
        class_order,
        expected_test_counts,
    )
    write_outputs(
        output_dir,
        leaf_table,
        plot_table,
        predictions,
        metrics,
        selected_features,
        dimensions,
        feature_columns,
        class_order,
        treatment_groups,
    )
    print("\nOut-of-fold metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
