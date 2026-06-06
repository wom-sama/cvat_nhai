import pytest

from cvat_nhai.models import BBox


def test_bbox_normalize_clamp_and_yolo() -> None:
    box = BBox(90, 80, 10, -5).normalized().clamp(100, 100)
    assert box == BBox(10, 0, 90, 80)
    assert box.is_valid()
    assert box.to_yolo(100, 100) == pytest.approx((0.5, 0.4, 0.8, 0.8))


def test_bbox_padding_is_clamped() -> None:
    box = BBox(5, 10, 25, 30).padded(0.5, 30, 40)
    assert box == BBox(0, 0, 30, 40)


def test_bbox_rejects_tiny_box() -> None:
    with pytest.raises(ValueError):
        BBox(1, 1, 2, 2).to_yolo(100, 100)
