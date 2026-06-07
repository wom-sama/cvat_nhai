import csv
from pathlib import Path

import pytest
from PIL import Image

from cvat_nhai.models import BBox, YoloAnnotation
from cvat_nhai.utils import atomic_write_yaml, load_yaml, names_from_yaml
from cvat_nhai.yolo_editor import (
    YoloDatasetEditor,
    YoloEditorError,
    export_classification_folder,
    read_yolo_annotations,
    scan_yolo_dataset,
    serialize_yolo_annotations,
)


NAMES = ("green", "ripe", "bad")


def make_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    atomic_write_yaml(
        root / "data.yaml",
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(NAMES),
            "names": list(NAMES),
        },
    )
    atomic_write_yaml(
        root / "canbang.yaml",
        {
            "dataset_balance": {
                "total_images": 2,
                "total_objects": 3,
                "classes": {
                    "0": {"count": 1},
                    "1": {"count": 1},
                    "2": {"count": 1},
                },
            }
        },
    )
    Image.new("RGB", (200, 100), "green").save(
        root / "images" / "train" / "a.jpg"
    )
    Image.new("RGB", (120, 120), "orange").save(
        root / "images" / "val" / "b.jpg"
    )
    (root / "labels" / "train" / "a.txt").write_text(
        "0 0.250000 0.500000 0.400000 0.600000\n"
        "1 0.750000 0.500000 0.400000 0.600000\n",
        encoding="utf-8",
    )
    (root / "labels" / "val" / "b.txt").write_text(
        "2 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )
    return root


def test_scan_and_round_trip_multiple_yolo_boxes(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)

    assert index.class_names == NAMES
    assert len(index.samples) == 2
    sample = index.samples[0]
    annotations = read_yolo_annotations(sample.label_path, 200, 100, 3)
    assert len(annotations) == 2
    assert annotations[0].bbox == BBox(10, 20, 90, 80)
    assert annotations[1].bbox == BBox(110, 20, 190, 80)

    serialized = serialize_yolo_annotations(annotations, 200, 100, 3)
    assert serialized == sample.label_path.read_text(encoding="utf-8")


def test_tiny_positive_bbox_is_loaded_and_round_trips(tmp_path: Path) -> None:
    label = tmp_path / "tiny.txt"
    original = (
        "4 0.032531 0.144258 0.002156 0.002172\n"
        "4 0.514641 0.505453 0.970719 0.715906\n"
    )
    label.write_text(original, encoding="utf-8")

    annotations = read_yolo_annotations(label, 640, 640, 5)

    assert len(annotations) == 2
    assert annotations[0].bbox.width == pytest.approx(1.37984)
    assert annotations[0].bbox.height == pytest.approx(1.39008)
    assert annotations[1].bbox.area > annotations[0].bbox.area
    assert serialize_yolo_annotations(annotations, 640, 640, 5) == original


def test_bbox_truly_outside_image_is_rejected(tmp_path: Path) -> None:
    label = tmp_path / "outside.txt"
    label.write_text(
        "0 0.950000 0.500000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    with pytest.raises(YoloEditorError, match="vuot ngoai bien"):
        read_yolo_annotations(label, 640, 640, 5)


def test_save_annotations_backs_up_and_updates_balance(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    editor = YoloDatasetEditor(index)
    sample = index.samples[0]
    changed = (
        YoloAnnotation(2, BBox(20, 10, 180, 90)),
    )

    editor.save_annotations(sample, changed, 200, 100)

    assert sample.label_path.read_text(encoding="utf-8").startswith(
        "2 0.500000 0.500000 0.800000 0.800000"
    )
    backups = list(
        (root / ".cvat_nhai_editor_archive" / "edits").rglob("a.txt")
    )
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8").count("\n") == 2
    balance = load_yaml(root / "canbang.yaml")["dataset_balance"]
    assert balance["total_objects"] == 2
    assert balance["classes"]["0"]["count"] == 0
    assert balance["classes"]["1"]["count"] == 0
    assert balance["classes"]["2"]["count"] == 2


def test_save_rolls_back_label_when_metadata_update_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    editor = YoloDatasetEditor(index)
    sample = index.samples[0]
    original = sample.label_path.read_text(encoding="utf-8")

    def fail_update(*args, **kwargs):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(editor, "_update_balance", fail_update)
    with pytest.raises(OSError, match="metadata unavailable"):
        editor.save_annotations(
            sample,
            (YoloAnnotation(2, BBox(20, 10, 180, 90)),),
            200,
            100,
        )

    assert sample.label_path.read_text(encoding="utf-8") == original


def test_delete_sample_archives_image_and_label(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    editor = YoloDatasetEditor(index)
    sample = index.samples[1]

    editor.delete_sample(sample, 120, 120)

    assert not sample.image_path.exists()
    assert not sample.label_path.exists()
    archive = root / ".cvat_nhai_editor_archive" / "deleted"
    assert len(list(archive.rglob("b.jpg"))) == 1
    assert len(list(archive.rglob("b.txt"))) == 1
    balance = load_yaml(root / "canbang.yaml")["dataset_balance"]
    assert balance["total_images"] == 1
    assert balance["total_objects"] == 2


def test_delete_rolls_back_files_when_metadata_update_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    editor = YoloDatasetEditor(index)
    sample = index.samples[1]

    def fail_update(*args, **kwargs):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(editor, "_update_balance", fail_update)
    with pytest.raises(OSError, match="metadata unavailable"):
        editor.delete_sample(sample, 120, 120)

    assert sample.image_path.exists()
    assert sample.label_path.exists()


def test_segmentation_rows_are_rejected_without_modification(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    label = root / "labels" / "train" / "a.txt"
    original = "0 0.1 0.1 0.2 0.2 0.3 0.3\n"
    label.write_text(original, encoding="utf-8")

    with pytest.raises(YoloEditorError, match="5 cot"):
        read_yolo_annotations(label, 200, 100, 3)
    assert label.read_text(encoding="utf-8") == original


def test_export_classification_folder_uses_edited_labels(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    editor = YoloDatasetEditor(index)
    first = index.samples[0]
    editor.save_annotations(
        first,
        (YoloAnnotation(2, BBox(50, 20, 150, 80)),),
        200,
        100,
    )
    destination = tmp_path / "classification"
    destination.mkdir()

    report = export_classification_folder(
        index,
        destination,
        crop_padding=0.0,
    )

    assert report.images == 2
    assert report.objects == 2
    assert report.class_counts == {2: 2}
    crops = sorted(destination.rglob("*.jpg"))
    assert len(crops) == 2
    with Image.open(
        destination / "train" / "bad" / "a_box000.jpg"
    ) as crop:
        assert crop.size == (100, 60)
    with (destination / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["class_name"] for row in rows} == {"bad"}
    assert tuple(
        names_from_yaml(load_yaml(destination / "data.yaml"))
    ) == NAMES
    for split in ("train", "val", "test"):
        for class_name in NAMES:
            assert (destination / split / class_name).is_dir()


def test_export_refuses_nonempty_destination(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    destination = tmp_path / "classification"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(YoloEditorError, match="phai rong"):
        export_classification_folder(index, destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == (
        "do not overwrite"
    )
