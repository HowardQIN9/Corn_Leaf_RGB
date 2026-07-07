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
