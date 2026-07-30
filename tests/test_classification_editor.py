import csv
from pathlib import Path

import pytest
from PIL import Image

from cvat_nhai.classification_editor import (
    ClassificationDatasetEditor,
    scan_classification_dataset,
)
from cvat_nhai.utils import atomic_write_yaml, load_yaml, names_from_yaml


NAMES = ("green", "ripe", "bad")


def _make_classification_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        for class_name in NAMES:
            (root / split / class_name).mkdir(parents=True)
    atomic_write_yaml(
        root / "data.yaml",
        {
            "format": "classification_folder",
            "path": ".",
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(NAMES),
            "names": {index: name for index, name in enumerate(NAMES)},
        },
    )
    train_green = root / "train" / "green" / "a.jpg"
    train_ripe = root / "train" / "ripe" / "already.jpg"
    val_bad = root / "val" / "bad" / "c.jpg"
    Image.new("RGB", (64, 64), "green").save(train_green)
    Image.new("RGB", (64, 64), "orange").save(train_ripe)
    Image.new("RGB", (64, 64), "red").save(val_bad)
    with (root / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_image",
                "output_image",
                "class_id",
                "class_name",
                "source",
                "source_split",
                "leakage_group",
            ]
        )
        writer.writerow(
            [
                "train",
                "raw/a.jpg",
                str(train_green),
                0,
                "green",
                0,
                "train",
                "g-a",
            ]
        )
        writer.writerow(
            [
                "train",
                "raw/already.jpg",
                str(train_ripe),
                1,
                "ripe",
                0,
                "train",
                "g-already",
            ]
        )
        writer.writerow(
            [
                "val",
                "raw/c.jpg",
                str(val_bad),
                2,
                "bad",
                0,
                "val",
                "g-c",
            ]
        )
    return root


def test_classification_editor_moves_file_and_updates_metadata(
    tmp_path: Path,
) -> None:
    root = _make_classification_dataset(tmp_path / "class_f")
    index = scan_classification_dataset(root)
    sample = next(
        item for item in index.samples if item.image_path.name == "a.jpg"
    )

    changed = ClassificationDatasetEditor(index).change_class(sample, 1)

    assert not sample.image_path.exists()
    assert changed.split == "train"
    assert changed.class_id == 1
    assert changed.image_path == root / "train" / "ripe" / "a.jpg"
    assert changed.image_path.exists()

    with (root / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    row = next(item for item in rows if item["source_image"] == "raw/a.jpg")
    assert row["output_image"] == str(changed.image_path)
    assert row["class_id"] == "1"
    assert row["class_name"] == "ripe"
    assert row["split"] == "train"
    assert row["leakage_group"] == "g-a"

    balance = load_yaml(root / "canbang.yaml")["dataset_balance"]
    assert balance["classes"]["0"]["count"] == 0
    assert balance["classes"]["1"]["count"] == 2
    assert balance["classes"]["2"]["count"] == 1
    assert balance["splits"]["train"]["classes"]["1"] == 2
    stats = load_yaml(root / "stats.json")
    assert stats["splits"]["train"]["classes"]["green"] == 0
    assert stats["splits"]["train"]["classes"]["ripe"] == 2
    assert tuple(names_from_yaml(load_yaml(root / "data.yaml"))) == NAMES
    assert load_yaml(root / "data.yaml")["image_size"] == [640, 640]


def test_classification_editor_deletes_to_archive_and_removes_manifest_row(
    tmp_path: Path,
) -> None:
    root = _make_classification_dataset(tmp_path / "class_f")
    index = scan_classification_dataset(root)
    sample = next(
        item for item in index.samples if item.image_path.name == "c.jpg"
    )

    archive = ClassificationDatasetEditor(index).delete_sample(sample)

    assert not sample.image_path.exists()
    assert archive.exists()
    assert ".cvat_nhai_classification_archive" in str(archive)
    with (root / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["source_image"] != "raw/c.jpg" for row in rows)
    balance = load_yaml(root / "canbang.yaml")["dataset_balance"]
    assert balance["total_objects"] == 2
    assert balance["classes"]["2"]["count"] == 0
    stats = load_yaml(root / "stats.json")
    assert stats["splits"]["val"]["images"] == 0
    assert stats["splits"]["val"]["classes"]["bad"] == 0


def test_classification_editor_rolls_back_move_when_metadata_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _make_classification_dataset(tmp_path / "class_f")
    index = scan_classification_dataset(root)
    editor = ClassificationDatasetEditor(index)
    sample = next(
        item for item in index.samples if item.image_path.name == "a.jpg"
    )
    original_manifest = (root / "manifest.csv").read_text(encoding="utf-8")

    def fail_refresh() -> None:
        raise OSError("metadata locked")

    monkeypatch.setattr(editor, "refresh_metadata", fail_refresh)
    with pytest.raises(OSError, match="metadata locked"):
        editor.change_class(sample, 1)

    assert sample.image_path.exists()
    assert not (root / "train" / "ripe" / "a.jpg").exists()
    assert (root / "manifest.csv").read_text(
        encoding="utf-8",
    ) == original_manifest


def test_classification_editor_creates_manifest_when_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "class_f"
    (root / "train" / "green").mkdir(parents=True)
    (root / "train" / "ripe").mkdir(parents=True)
    atomic_write_yaml(
        root / "data.yaml",
        {
            "format": "classification_folder",
            "names": ["green", "ripe"],
        },
    )
    image_path = root / "train" / "green" / "new.jpg"
    Image.new("RGB", (32, 32), "green").save(image_path)
    index = scan_classification_dataset(root)

    changed = ClassificationDatasetEditor(index).change_class(
        index.samples[0],
        1,
    )

    assert changed.image_path == root / "train" / "ripe" / "new.jpg"
    with (root / "manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "split": "train",
            "source_image": "",
            "output_image": str(changed.image_path),
            "class_id": "1",
            "class_name": "ripe",
            "source": "",
            "source_split": "train",
            "leakage_group": "",
        }
    ]
