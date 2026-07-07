import numpy as np

from leafseg.config import SegmentationConfig
from leafseg.morphology import keep_largest_component
from leafseg.segmentation import segment_leaf_hsv


def test_hsv_segmentation_outputs_binary_values() -> None:
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    image[4:12, 5:15] = np.array([40, 160, 40], dtype=np.uint8)

    mask = segment_leaf_hsv(image, SegmentationConfig(min_leaf_area_pixels=1))

    assert set(np.unique(mask)).issubset({0, 255})
    assert mask.dtype == np.uint8


def test_keep_largest_component_keeps_largest_region() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[8:16, 8:18] = 255

    largest = keep_largest_component(mask)

    assert np.count_nonzero(largest) == 80
    assert largest[10, 10] == 255
    assert largest[1, 1] == 0
