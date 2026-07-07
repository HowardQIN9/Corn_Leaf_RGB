import numpy as np

from leafseg.config import BoundingBox
from leafseg.morphology import crop_by_bbox, get_bbox


def test_get_bbox_with_padding_clipped_to_image() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:8] = 255

    bbox = get_bbox(mask, padding=2, image_shape=mask.shape)

    assert bbox == BoundingBox(x=1, y=0, width=9, height=7)


def test_crop_by_bbox_returns_expected_shape() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    bbox = BoundingBox(x=5, y=4, width=10, height=6)

    crop = crop_by_bbox(image, bbox)

    assert crop.shape == (6, 10, 3)
