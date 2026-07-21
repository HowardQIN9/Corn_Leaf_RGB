# Corn Leaf Segmentation

Clean, maintainable Python pipeline for segmenting individual corn leaves from TIFF/JPEG/PNG images taken on a mostly white backlight/background.

The primary analysis outputs are the original image, a full-size binary leaf mask, and metadata. Crops, overlays, and segmented previews are convenience outputs for review and downstream processing.

## Architecture

The code is organized as a package so each stage can be reused by later modules:

- `leafseg.config`: typed configuration with JSON/YAML loading.
- `leafseg.io`: image reading, 8-bit RGB preview creation, and output writers.
- `leafseg.segmentation`: HSV segmentation by default, with optional ExG.
- `leafseg.morphology`: cleanup, component filtering, hole filling, bbox, and crops.
- `leafseg.qc`: QC flags and overlay generation.
- `leafseg.metadata`: CSV row construction and batch metadata writing.
- `leafseg.pipeline`: one-image and folder orchestration.
- `leafseg.cli`: command-line interface.

This structure keeps the scientific data path explicit: original pixels are preserved as closely as possible for TIFF crops, masks are binary PNGs, and previews are only visualization. Downstream leaf alignment, midrib detection, transverse profile extraction, and color/texture feature extraction can depend on stable mask, bbox, crop, and metadata outputs without depending on CLI code or QC visualization code.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Recommended Workflow

1. Copy 20-30 representative images into `data/test_subset/`.
2. Run the pipeline on `data/test_subset/`.
3. Inspect `outputs/test_run/qc/` overlay images.
4. Adjust global parameters if needed.
5. Run the same fixed parameters on the full image folder.
6. Use `outputs/test_run/metadata/segmentation_metadata.csv` to find abnormal images for manual review.

## Example Command

```powershell
python -m leafseg.cli `
  --input_dir data/test_subset `
  --output_dir outputs/test_run `
  --method hsv `
  --h_min 20 `
  --h_max 95 `
  --s_min 35 `
  --v_min 25 `
  --bbox_padding 20 `
  --morph_kernel_size 21 `
  --save_segmented_preview false
```

Equivalent script entry point:

```powershell
python scripts/segment_leaves.py --input_dir data/test_subset --output_dir outputs/test_run
```

## Output Structure

```text
output_dir/
  masks/
    IMG_001_leaf_mask.png
  crops/
    IMG_001_leaf_crop.tif
    IMG_001_leaf_crop_mask.png
  qc/
    IMG_001_overlay.png
  metadata/
    segmentation_metadata.csv
  segmented_preview/
    IMG_001_segmented_preview.png
```

## Configuration File

JSON files are supported by default. YAML files are supported when `PyYAML` is installed.

```json
{
  "method": "hsv",
  "h_min": 20,
  "h_max": 95,
  "s_min": 35,
  "v_min": 25,
  "morph_kernel_size": 21,
  "bbox_padding": 20,
  "save_segmented_preview": false
}
```

Then run:

```powershell
python -m leafseg.cli --input_dir data/test_subset --output_dir outputs/test_run --config config.json
```

CLI arguments override config-file values.

## Testing

```powershell
pytest
```

## Geometric Centerline And Sampling Lines

After segmentation, you can generate horizontal geometric centerlines and perpendicular sampling lines from binary masks. The centerline is computed from mask geometry only:

```text
y_center_horizontal = median((y_top(x) + y_bottom(x)) / 2)
```

It is a proxy for the leaf direction, not a detected biological midrib. By default, the module creates 5 vertical sampling lines at 5 interior positions along the leaf, extracts green-channel profiles along those lines, and averages each profile point across 5 pixels parallel to the horizontal centerline.

For the Leaf2-only crop masks created by `scripts/extract_leaf2_crops.py`:

```powershell
python -m leafsampling.cli `
  --mask_dir outputs/RGB_tall_v9_leaf2_crops `
  --image_dir outputs/RGB_tall_v9_leaf2_crops `
  --output_dir outputs/RGB_tall_v9_leaf2_centerline_sampling `
  --n_sampling_lines 5 `
  --strip_width_px 5 `
  --smoothing_window_length 51 `
  --smoothing_polyorder 2 `
  --edge_trim_ratio 0.05 `
  --tip_trim_ratio 0.02 `
  --min_leaf_width 20
```

Outputs:

```text
output_dir/
  centerline/
    IMG_001_centerline.csv
  sampling_lines/
    IMG_001_sampling_lines.csv
  profiles/
    IMG_001_green_profiles.csv
  qc/
    IMG_001_centerline_sampling_overlay.png
  metadata/
    centerline_sampling_metadata.csv
```

The sampled green-channel curves are saved in two forms:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/
  profiles/
    IMG_001_green_profiles.csv       # one long-format curve table per leaf
  metadata/
    all_leaf2_green_profiles.csv     # all per-leaf curve tables combined
```

Each leaf has 5 transverse curves, identified by `sample_id`. Within each
curve, `position_fraction` runs from one leaf edge to the other, and
`green_mean` is the green-channel intensity averaged across the configured
5-pixel-wide strip. The per-leaf profile CSVs are created by
`python -m leafsampling.cli`. Combine them for downstream analysis with:

```powershell
python scripts/combine_green_profiles.py
```

## Midrib Valley Detection

After green profiles are combined, detect the midrib-related feature in the
middle of each transverse profile:

```powershell
python scripts/detect_midrib_peaks.py
```

The current script defaults to detecting a broad bright peak in `green_mean`
(`--peak_polarity bright`). It searches only the middle profile region,
rejects narrow spikes, and marks a leaf as `pass` only when at least 3 of 5
profile lines have consistent detected positions. Use
`--peak_polarity dark` when the image convention makes the midrib a dark
valley instead.

Outputs:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/
  midrib_peak_line_results.csv
  midrib_peak_leaf_summary.csv
  all_leaf2_green_profiles_with_midrib_sides.csv
```

The annotated profile table adds `side_of_midrib`, `distance_from_midrib_split`, and `is_midrib_peak_region` for downstream analysis.

To visually inspect the detected valleys and split points:

```powershell
python scripts/plot_midrib_detection.py
```

This writes one PNG per leaf to:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/qc_plots/
```

## Split And Separated Curves

After automatic detection and any manual adjustment, split each transverse
curve at the two boundaries of the detected midrib region:

```powershell
python scripts/split_profiles_from_midrib.py
```

The default command reads the manually reviewed line results and writes:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/
  midrib_peak_line_results_manual_adjusted.csv
  all_leaf2_green_profiles_split_from_midrib.csv
  split_profile_plots/
```

In the split table, each original curve becomes an `upper` and `lower` curve.
`relative_distance_from_midrib` starts at 0 at the midrib boundary and reaches
1 at the corresponding leaf edge.

Separate each split curve into a smooth lower-envelope mesophyll baseline and
a residual peak curve with:

```powershell
python scripts/separate_split_curves.py `
  --valley_distance 7 `
  --smooth_window 21
```

The current valley-anchor distance is 7 profile points. Smaller values allow
the baseline to recognize the bottoms around smaller, closely spaced peaks;
larger values ignore more local valleys. The mesophyll-envelope smoothing
window is 21 profile points and controls the smoothness after those valley
anchors are connected.

Final curve outputs are saved here:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/midrib_detection/
  all_leaf2_green_profiles_split_meso_peak.csv
  split_curve_meso_plots/             # raw curve plus mesophyll baseline
  split_curve_meso_only_plots/        # mesophyll baseline only, meso-scaled y-axis
  split_curve_separation_plots/       # residual peak curve
```

The final CSV retains the original `green_mean` curve and adds:

- `green_mean_meso`: smoothed lower-envelope baseline.
- `green_mean_peak`: residual curve, calculated as
  `green_mean - green_mean_meso`.

## Curve Feature Analysis

Extract modeling features from the final split curves with:

```powershell
python scripts/extract_curve_features.py
```

The default run reads
`midrib_detection/all_leaf2_green_profiles_split_meso_peak.csv` and writes:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/feature_analysis/
  leaf2_green_curve_features.csv   # one row per sample_id and upper/lower side
  leaf2_green_image_features.csv   # one row per image, aggregated across curves
```

Every half-curve is first interpolated onto 256 evenly spaced points from the
midrib boundary (`relative_distance_from_midrib = 0`) to the leaf edge (`1`).
This makes length-dependent features comparable between leaves. Features are
calculated for the complete half-curve (`full`) and, by default, three equal
distance zones:

- `proximal`: 0 to 1/3, closest to the midrib.
- `middle`: 1/3 to 2/3.
- `distal`: 2/3 to 1, closest to the leaf edge.

The module adapts the previous slope-feature workflow to the current
long-format profile table. It calculates:

- Signal distribution and scale: mean, median, standard deviation, variance,
  range, quantiles, IQR, RMS, energy, and absolute-value summaries.
- Direction and variability: positive/negative fractions and sums,
  zero-crossing rate, roughness, high-amplitude fraction, AUC, and linear
  slope.
- Mesophyll shape: first derivative (`d1`), second derivative (`d2`), and
  geometric curvature `abs(d2) / (1 + d1^2)^1.5` from `green_mean_meso`.
- Residual peaks: count, density, height, prominence, and width from
  `green_mean_peak`.
- Curve QC descriptors: original point count and normalized-distance coverage.

Feature names follow this pattern:

```text
value_column__zone__family_feature
green_mean_meso__middle__d1_mean
green_mean_meso__full__curvature_q95
green_mean_peak__proximal__peaks_prominence_mean
```

The curve-level table keeps `sample_id` and `midrib_side`, so it can be
filtered to compare the ten fixed curve positions using the same model and
data folds. The image-level table aggregates numeric curve features with
mean, standard deviation, minimum, and maximum by default.

Common options include:

```powershell
python scripts/extract_curve_features.py `
  --n_zones 3 `
  --resample_points 256 `
  --derivative_column green_mean_meso `
  --peak_prominence_fraction 0.10 `
  --image_aggregations mean std min max
```

Use `--skip_image_summary` to write only the curve-level table. Additional
intensity or vegetation-index columns can be analyzed with `--value_columns`
after they have been added to the long-format input CSV.

Feature selection and model tuning must be performed inside each training
fold. If multiple plant images share one plot-level reference measurement,
split cross-validation by `plot_number`; do not randomly split those images
between training and validation data.

## Leakage-Safe Treatment Classification

Run plot-level feature selection and seven-class treatment classification with:

```powershell
python scripts/select_features_classify_treatment.py
```

Configuration constants near the top of the script define the treatment,
block, plot, leaf, curve, and external label columns. The defaults match the
current curve-level output:

- Treatment: `treatment`, joined from `Treatment` in the label workbook's
  `Tall` sheet.
- Block: `block`, derived from the hundreds digit of `plot_number`.
- Plot ID: `plot_number`.
- Leaf ID: `source_profile_file`.
- Curve ID: `sample_id` plus `midrib_side`.

The script takes the median of ten curves per leaf and then the median of five
leaves per plot. It verifies the balanced 7-treatment by 3-block design before
running leave-one-block-out validation. Within every training fold it fits
missing-value imputation, constant/near-zero-variance filtering, complete-
linkage Pearson redundancy clustering at `|r| >= 0.95`, ANOVA Top 20, greedy
mRMR Top 3, and scaling. None of those fitted transformations sees the held-
out block.

Three classifiers are reported from the same out-of-fold splits: L2
multinomial logistic regression, nearest centroid, and PLS-DA with at most two
components. CSV results and PNG figures are written to:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/treatment_classification/
```

The PLS-DA score plot uses all plot rows and the most frequently selected
features only for visualization. It is explicitly labeled exploratory and is
not used to calculate validation metrics.

To group the ordered N rates into three broader classes while preserving the
same leakage-safe workflow, run:

```powershell
python scripts/select_features_classify_treatment.py --three_class
```

The fixed grouping is `Low={0,60}`, `Medium={85,120,153}`, and
`High={180,240}`. Results are written separately to
`treatment_classification_3class/`, so the original seven-class analysis is
not overwritten.

## Independent Image-Representation Experiment

Compare five image representations without combining their features:

```powershell
python scripts/run_image_representation_experiment.py
```

The experiment holds the existing leaf masks, five transverse sampling lines,
and manually adjusted midrib boundaries fixed. For every representation it
independently samples ten half-curves per leaf, calculates the same 696 curve
features, aggregates curves to leaves and leaves to plots, and runs the same
three-class leave-one-block-out analysis.

The five separate pipelines are:

- `raw_green`: original green-channel values.
- `ngrdi`: `(G-R)/(G+R+1/255)` from RGB scaled to 0-1.
- `exg`: `2G-R-B` from RGB scaled to 0-1.
- `g2_rb`: `G^2 / ((R+1/255)(B+1/255))`; the fixed one-count
  pseudocount prevents division by zero in dark blue-channel pixels.
- `clahe_green`: mask-aware CLAHE on green, using normalized clip limit 0.01
  and an 8 by 8 tile grid. Mask-exterior pixels are median-filled only for the
  CLAHE operation and are discarded before profile sampling.

All calculations and profile averages use leaf-mask pixels only. The PNGs in
each `qc_examples/` directory use within-image percentile scaling strictly for
visual inspection; that visualization scaling is not used for modeling.

Outputs are organized as:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/image_representation_experiment/
  raw_green/
  ngrdi/
  exg/
  g2_rb/
  clahe_green/
    feature_analysis/curve_features.csv
    treatment_classification_3class/
    qc_examples/
  comparison/
    representation_model_metrics.csv
    representation_metrics_wide.csv
    representation_best_models.csv
    controlled_variables.csv
    plot_treatment_mapping_used.csv
    representation_balanced_accuracy.png
  experiment_manifest.json
```

The comparison table reports all three models for every representation and a
three-class chance reference of 1/3. Each representation is modeled alone;
no representation-level feature tables are concatenated.

## NGRDI Best Two-Curve Experiment

To compare all 45 ways to retain two of the ten NGRDI half-curves, run:

```powershell
python scripts/select_best_two_ngrdi_curves.py
```

The ten positions are `line_1_upper`, `line_1_lower`, through
`line_5_upper`, `line_5_lower`. A selected pair is fixed across all leaves;
two curve rows are median-aggregated per leaf, then five leaves are
median-aggregated per plot. Curves and leaves are never treated as independent
samples.

Two results are deliberately separated:

- The exploratory fixed-pair ranking evaluates all 45 pairs using the same
  three LOBO folds. It identifies promising physical sampling positions, but
  its best accuracy is optimistic because the pair is chosen after comparing
  all 45 OOF results.
- The nested result chooses the curve pair using only the two outer-training
  blocks before predicting the held-out block. This is the leakage-safe
  performance estimate for the pair-selection procedure.

Results are saved in:

```text
outputs/RGB_tall_v9_leaf2_centerline_sampling/ngrdi_two_curve_pair_experiment/
  exploratory_best_pair_by_model.csv
  exploratory_best_pair_oof_predictions.csv
  exploratory_best_pair_confusion/
  best_pair_selected_feature_frequency.csv
  best_pair_stable_feature_plot_values.csv
  best_pair_stable_feature_plot.png
  stable_feature_only_oof_metrics.csv
  exploratory_pair_model_metrics.csv
  exploratory_pair_overall_ranking.csv
  nested_selected_pairs_by_outer_fold.csv
  nested_oof_metrics.csv
  comparison_all10_vs_two_curve.csv
  exploratory_pair_balanced_accuracy_heatmaps.png
  exploratory_top_pairs.png
```
