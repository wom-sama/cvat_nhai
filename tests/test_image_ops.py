from PIL import Image

from cvat_nhai.image_ops import (
    CLASSIFICATION_IMAGE_SIZE,
    CLASSIFICATION_PADDING_COLOR,
    classification_crop,
)
from cvat_nhai.models import BBox


def test_classification_crop_letterboxes_to_640_without_distortion() -> None:
    image = Image.new("RGB", (200, 100), (220, 40, 20))

    result = classification_crop(
        image,
        BBox(20, 10, 180, 90),
        crop_padding=0.1,
    )

    assert result.size == (
        CLASSIFICATION_IMAGE_SIZE,
        CLASSIFICATION_IMAGE_SIZE,
    )
    assert result.getpixel((320, 320)) == (220, 40, 20)
    assert result.getpixel((0, 0)) == CLASSIFICATION_PADDING_COLOR


def test_classification_crop_handles_subpixel_bbox() -> None:
    image = Image.new("RGB", (20, 20), "green")

    result = classification_crop(
        image,
        BBox(5.1, 5.1, 5.4, 5.4),
        crop_padding=0.0,
    )

    assert result.size == (640, 640)
