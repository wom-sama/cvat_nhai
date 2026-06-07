import math

from PIL import Image, ImageOps

from .models import BBox


CLASSIFICATION_IMAGE_SIZE = 640
CLASSIFICATION_PADDING_COLOR = (114, 114, 114)


def classification_crop(
    image: Image.Image,
    bbox: BBox,
    crop_padding: float,
    output_size: int = CLASSIFICATION_IMAGE_SIZE,
) -> Image.Image:
    if output_size <= 0:
        raise ValueError("Classification output size must be positive")
    width, height = image.size
    padded = bbox.padded(crop_padding, width, height)
    left = max(0, min(width - 1, math.floor(padded.x1)))
    top = max(0, min(height - 1, math.floor(padded.y1)))
    right = max(left + 1, min(width, math.ceil(padded.x2)))
    bottom = max(top + 1, min(height, math.ceil(padded.y2)))
    crop = image.crop((left, top, right, bottom))
    return ImageOps.pad(
        crop,
        (output_size, output_size),
        method=Image.Resampling.LANCZOS,
        color=CLASSIFICATION_PADDING_COLOR,
        centering=(0.5, 0.5),
    )
