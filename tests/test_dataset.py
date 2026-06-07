import csv
from pathlib import Path

import pytest
from PIL import Image

from cvat_nhai.constants import CLASS_NAMES
from cvat_nhai.dataset import DatasetManager
from cvat_nhai.dataset import DatasetError
from cvat_nhai.models import BBox, DatasetPaths
from cvat_nhai.schema import initialize_empty_datasets
from cvat_nhai.utils import atomic_write_yaml, load_yaml, names_from_yaml


def make_manager(tmp_path: Path) -> tuple:
    detection = tmp_path / "detection"
    classification = tmp_path / "classification"
    archive = tmp_path / "archive"
    initialize_empty_datasets(detection, classification)
    manager = DatasetManager(
        DatasetPaths(detection, classification, archive),
        crop_padding=0.1,
    )
    return manager, detection, classification, archive


def test_annotation_writes_both_datasets_and_undoes(tmp_path: Path) -> None:
    manager, detection, classification, archive = make_manager(tmp_path)
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source = source_dir / "mango.jpg"
    Image.new("RGB", (200, 100), "#CCAA33").save(source)

    result = manager.annotate(
        source,
        BBox(20, 10, 180, 90),
        class_id=1,
        requested_split="val",
    )

    assert not source.exists()
    assert result.detection_image is not None
    assert result.detection_image.exists()
    assert result.detection_label is not None
    assert result.detection_label.read_text(encoding="utf-8").startswith(
        "1 0.500000 0.500000 0.800000 0.800000"
    )
    assert result.classification_crop is not None
    assert result.classification_crop.exists()
    with Image.open(result.classification_crop) as crop:
        assert crop.size == (640, 640)
        corner = crop.getpixel((0, 0))
        assert all(
            abs(channel - 114) <= 3 for channel in corner
        )

    with (classification / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["class_name"] == CLASS_NAMES[1]
    assert rows[0]["source_image"] == str(result.detection_image)

    restored = manager.undo_latest()
    assert restored == source
    assert source.exists()
    assert not result.detection_image.exists()
    assert not result.detection_label.exists()
    assert not result.classification_crop.exists()


def test_reject_moves_to_archive_and_can_undo(tmp_path: Path) -> None:
    manager, _, _, _ = make_manager(tmp_path)
    source = tmp_path / "incoming.jpg"
    Image.new("RGB", (30, 30), "red").save(source)

    result = manager.reject(source)
    assert not source.exists()
    assert result.archived_source is not None
    assert result.archived_source.exists()

    restored = manager.undo_latest()
    assert restored == source
    assert source.exists()


def test_initialized_yaml_uses_five_classes(tmp_path: Path) -> None:
    _, detection, classification, _ = make_manager(tmp_path)
    assert tuple(names_from_yaml(load_yaml(detection / "data.yaml"))) == CLASS_NAMES
    assert (
        tuple(names_from_yaml(load_yaml(classification / "data.yaml")))
        == CLASS_NAMES
    )
    classification_yaml = load_yaml(classification / "data.yaml")
    assert classification_yaml["image_size"] == [640, 640]
    assert classification_yaml["resize_mode"] == "letterbox"
    stats = load_yaml(classification / "stats.json")
    assert stats["output_size"] == [640, 640]
    assert stats["resize_mode"] == "letterbox"


def test_existing_classification_yaml_gets_resize_metadata(
    tmp_path: Path,
) -> None:
    manager, _, classification, _ = make_manager(tmp_path)
    payload = load_yaml(classification / "data.yaml")
    payload.pop("image_size")
    payload.pop("resize_mode")
    atomic_write_yaml(classification / "data.yaml", payload)
    source = tmp_path / "existing_config.jpg"
    Image.new("RGB", (80, 60), "orange").save(source)

    manager.annotate(source, BBox(10, 10, 70, 50), 0, "train")

    upgraded = load_yaml(classification / "data.yaml")
    assert upgraded["image_size"] == [640, 640]
    assert upgraded["resize_mode"] == "letterbox"


def test_same_source_name_never_overwrites_outputs(tmp_path: Path) -> None:
    manager, _, _, _ = make_manager(tmp_path)
    first_dir = tmp_path / "incoming_a"
    second_dir = tmp_path / "incoming_b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "same.jpg"
    second = second_dir / "same.jpg"
    Image.new("RGB", (40, 40), "red").save(first)
    Image.new("RGB", (40, 40), "blue").save(second)

    first_result = manager.annotate(first, BBox(5, 5, 35, 35), 0, "train")
    second_result = manager.annotate(second, BBox(5, 5, 35, 35), 0, "train")

    assert first_result.detection_image != second_result.detection_image
    assert first_result.detection_image.exists()
    assert second_result.detection_image.exists()


def test_refuses_to_process_images_inside_output_dataset(tmp_path: Path) -> None:
    manager, detection, _, _ = make_manager(tmp_path)
    source = detection / "images" / "train" / "already_output.jpg"
    Image.new("RGB", (40, 40), "red").save(source)

    with pytest.raises(DatasetError):
        manager.annotate(source, BBox(5, 5, 35, 35), 0, "train")


def test_auto_split_is_stable(tmp_path: Path) -> None:
    manager, _, _, _ = make_manager(tmp_path)
    source = tmp_path / "incoming" / "stable.jpg"
    source.parent.mkdir()
    source.write_bytes(b"not loaded in this test")
    assert manager.choose_split(source) == manager.choose_split(source)
    assert manager.choose_split(tmp_path / "val" / "image.jpg") == "val"
