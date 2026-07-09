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

## Midrib Valley Detection

After green profiles are combined, detect the midrib-related valley in the middle of each transverse profile:

```powershell
python scripts/detect_midrib_peaks.py
```

The detector searches only the middle profile region, defaults to detecting a dark valley in `green_mean`, rejects narrow spikes, and marks a leaf as `pass` only when at least 3 of 5 profile lines have consistent detected positions.

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
