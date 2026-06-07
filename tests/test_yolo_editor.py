import csv
import threading
from pathlib import Path

import pytest
from PIL import Image

from cvat_nhai.models import BBox, YoloAnnotation
from cvat_nhai.utils import atomic_write_yaml, load_yaml, names_from_yaml
from cvat_nhai.yolo_editor import (
    YoloDatasetEditor,
    YoloEditorError,
    ExportCancelled,
    export_rebalanced_datasets,
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


def test_export_rebalanced_datasets_uses_edited_labels(
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

    report = export_rebalanced_datasets(
        index,
        destination,
        crop_padding=0.0,
    )

    assert report.images == 2
    assert report.objects == 2
    assert report.class_counts == {2: 2}
    classification = destination / "class_f"
    detection = destination / "yolo_f"
    crops = sorted(classification.rglob("*_box*.jpg"))
    assert len(crops) == 2
    a_crop = next(path for path in crops if path.name == "a_box000.jpg")
    with Image.open(a_crop) as crop:
        assert crop.size == (640, 640)
    with (classification / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["class_name"] for row in rows} == {"bad"}
    assert tuple(
        names_from_yaml(load_yaml(classification / "data.yaml"))
    ) == NAMES
    assert tuple(
        names_from_yaml(load_yaml(detection / "data.yaml"))
    ) == NAMES
    classification_yaml = load_yaml(classification / "data.yaml")
    assert classification_yaml["image_size"] == [640, 640]
    assert classification_yaml["resize_mode"] == "letterbox"
    classification_stats = load_yaml(classification / "stats.json")
    assert classification_stats["output_size"] == [640, 640]
    assert classification_stats["resize_mode"] == "letterbox"
    assert len(list(detection.rglob("*.txt"))) == 2
    assert all(
        line.startswith("2 ")
        for label in detection.rglob("*.txt")
        for line in label.read_text(encoding="utf-8").splitlines()
    )
    for split in ("train", "val", "test"):
        for class_name in NAMES:
            assert (classification / split / class_name).is_dir()
        assert (detection / "images" / split).is_dir()
        assert (detection / "labels" / split).is_dir()


def test_export_stratifies_each_class_and_groups_duplicates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "large_dataset"
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

    first_image = None
    for class_id in range(len(NAMES)):
        for item_index in range(10):
            source_split = ("train", "val", "test")[item_index % 3]
            image_path = (
                root
                / "images"
                / source_split
                / "c{}_{}.jpg".format(class_id, item_index)
            )
            image = Image.new("RGB", (32, 32))
            image.putdata(
                [
                    (
                        (x * 17 + item_index * 31 + class_id * 13) % 256,
                        (y * 29 + item_index * 11 + class_id * 47) % 256,
                        (
                            x * 7
                            + y * 19
                            + item_index * 23
                            + class_id * 59
                        )
                        % 256,
                    )
                    for y in range(32)
                    for x in range(32)
                ]
            )
            image.save(image_path, quality=98)
            image_path.with_suffix(".txt").write_text(
                "{} 0.500000 0.500000 0.750000 0.750000\n".format(
                    class_id
                ),
                encoding="utf-8",
            )
            label_path = (
                root
                / "labels"
                / source_split
                / image_path.with_suffix(".txt").name
            )
            image_path.with_suffix(".txt").replace(label_path)
            if class_id == 0 and item_index == 0:
                first_image = image_path

    duplicate = root / "images" / "test" / "duplicate_visual.jpg"
    with Image.open(first_image) as source_image:
        source_image.save(duplicate, quality=72)
    (root / "labels" / "test" / "duplicate_visual.txt").write_text(
        "0 0.500000 0.500000 0.750000 0.750000\n",
        encoding="utf-8",
    )

    destination = tmp_path / "balanced_export"
    report = export_rebalanced_datasets(
        scan_yolo_dataset(root),
        destination,
        crop_padding=0.0,
    )

    assert report.images == 31
    with (destination / "class_f" / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    per_class = {
        class_name: {
            split: sum(
                row["class_name"] == class_name
                and row["split"] == split
                for row in rows
            )
            for split in ("train", "val", "test")
        }
        for class_name in NAMES
    }
    for class_name, counts in per_class.items():
        total = sum(counts.values())
        assert all(counts[split] > 0 for split in counts), class_name
        assert abs(counts["train"] / total - 0.7) <= 0.16
        assert abs(counts["val"] / total - 0.2) <= 0.11
        assert abs(counts["test"] / total - 0.1) <= 0.11

    group_splits = {}
    for row in rows:
        group_splits.setdefault(row["leakage_group"], set()).add(
            row["split"]
        )
    assert all(len(splits) == 1 for splits in group_splits.values())
    duplicate_rows = [
        row
        for row in rows
        if Path(row["source_image"]).name
        in {"c0_0.jpg", "duplicate_visual.jpg"}
    ]
    assert len(duplicate_rows) == 2
    assert len({row["split"] for row in duplicate_rows}) == 1
    assert len({row["leakage_group"] for row in duplicate_rows}) == 1

    with (destination / "yolo_f" / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        detection_rows = list(csv.DictReader(handle))
    detection_assignments = {
        row["source_image"]: (
            row["split"],
            row["leakage_group"],
        )
        for row in detection_rows
    }
    assert len(detection_assignments) == 31
    assert all(
        detection_assignments[row["source_image"]]
        == (row["split"], row["leakage_group"])
        for row in rows
    )
    assert (
        load_yaml(destination / "yolo_f" / "data.yaml")[
            "split_strategy"
        ]
        == "stratified_group"
    )
    assert load_yaml(
        destination / "class_f" / "stats.json"
    )["source_groups"] == 30

    second_destination = tmp_path / "balanced_export_again"
    export_rebalanced_datasets(
        scan_yolo_dataset(root),
        second_destination,
        crop_padding=0.0,
    )
    with (second_destination / "yolo_f" / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        repeated_rows = list(csv.DictReader(handle))
    assert {
        row["source_image"]: row["split"] for row in repeated_rows
    } == {
        row["source_image"]: row["split"] for row in detection_rows
    }


def test_export_keeps_negative_images_in_yolo_dataset(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    negative = root / "images" / "test" / "negative.jpg"
    Image.new("RGB", (80, 60), "black").save(negative)
    (root / "labels" / "test" / "negative.txt").write_text(
        "",
        encoding="utf-8",
    )

    destination = tmp_path / "export"
    report = export_rebalanced_datasets(
        scan_yolo_dataset(root),
        destination,
        crop_padding=0.0,
    )

    assert report.images == 3
    assert report.objects == 3
    with (destination / "yolo_f" / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    negative_row = next(
        row
        for row in rows
        if Path(row["source_image"]).name == negative.name
    )
    negative_label = Path(negative_row["output_label"])
    assert negative_label.exists()
    assert negative_label.read_text(encoding="utf-8") == ""
    detection_balance = load_yaml(
        destination / "yolo_f" / "canbang.yaml"
    )["dataset_balance"]
    classification_balance = load_yaml(
        destination / "class_f" / "canbang.yaml"
    )["dataset_balance"]
    assert detection_balance["total_images"] == 3
    assert classification_balance["total_images"] == 2
    with (destination / "class_f" / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        crop_rows = list(csv.DictReader(handle))
    assert len(crop_rows) == 3


def test_export_refuses_nonempty_destination(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    destination = tmp_path / "classification"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(YoloEditorError, match="phai rong"):
        export_rebalanced_datasets(index, destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == (
        "do not overwrite"
    )


def test_export_refuses_destination_inside_source_dataset(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    index = scan_yolo_dataset(root)
    destination = root / "new_export"
    destination.mkdir()

    with pytest.raises(YoloEditorError, match="nam ngoai"):
        export_rebalanced_datasets(index, destination)
    assert not any(destination.iterdir())


def test_export_reports_progress_and_creates_named_subfolders(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    destination = tmp_path / "export"
    events = []

    report = export_rebalanced_datasets(
        scan_yolo_dataset(root),
        destination,
        progress_callback=events.append,
    )

    assert report.images == 2
    assert {
        path.name for path in destination.iterdir()
    } == {"yolo_f", "class_f"}
    assert (destination / "yolo_f" / "data.yaml").exists()
    assert (destination / "class_f" / "data.yaml").exists()
    assert events[0]["stage"] == "Dang chuan bi export"
    assert events[-1]["stage"] == "Export hoan tat"
    assert events[-1]["value"] == events[-1]["maximum"]
    assert {
        event["stage"].split(" ")[0]
        for event in events
        if event["stage"][0].isdigit()
    } == {"1/4", "2/4", "3/4", "4/4"}


def test_cancelled_export_removes_staging_and_keeps_destination_empty(
    tmp_path: Path,
) -> None:
    root = make_yolo_dataset(tmp_path / "dataset")
    destination = tmp_path / "export"
    destination.mkdir()
    cancel_event = threading.Event()

    def cancel_after_first_image(payload):
        if payload["stage"].startswith("1/4"):
            cancel_event.set()

    with pytest.raises(ExportCancelled):
        export_rebalanced_datasets(
            scan_yolo_dataset(root),
            destination,
            progress_callback=cancel_after_first_image,
            cancel_event=cancel_event,
        )

    assert not any(destination.iterdir())
    assert not list(
        tmp_path.glob(".cvat_nhai_export_*")
    )
