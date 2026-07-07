import numpy as np

from leafseg.config import BoundingBox, SegmentationConfig, SegmentationResult
from leafseg.metadata import REQUIRED_METADATA_COLUMNS, result_to_metadata_row


def test_metadata_row_contains_required_fields() -> None:
    result = SegmentationResult(
        filename="IMG_001.tif",
        mask=np.zeros((10, 10), dtype=np.uint8),
        bbox=BoundingBox(1, 2, 3, 4),
        leaf_area_pixels=12,
        mask_area_ratio=0.12,
        num_connected_components_before_filtering=2,
        qc_flag="pass",
        notes="",
    )

    row = result_to_metadata_row(result, SegmentationConfig(), (10, 10))

    assert set(REQUIRED_METADATA_COLUMNS).issubset(row.keys())
    assert row["bbox_x"] == 1
    assert row["threshold_method"] == "hsv"
