import numpy as np

from leafseg.io import make_rgb_8bit


def test_make_rgb_8bit_handles_uint16_grayscale() -> None:
    image = np.array([[0, 65535]], dtype=np.uint16)

    rgb = make_rgb_8bit(image)

    assert rgb.shape == (1, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 1, 0] == 255
