import csv
from pathlib import Path

from PIL import Image

from cvat_nhai.constants import CLASS_NAMES
from cvat_nhai.migration import apply_migration, plan_migration
from cvat_nhai.utils import load_yaml, names_from_yaml


def _write_manifest(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_image",
                "output_image",
                "class_id",
                "class_name",
                "source",
            ]
        )
        writer.writerows(rows)


def test_plan_and_apply_four_to_five_class_migration(tmp_path: Path) -> None:
    detection = tmp_path / "detection"
    classification = tmp_path / "classification"
    images = detection / "images" / "train"
    labels = detection / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    image_a = images / "a.jpg"
    image_b = images / "b.jpg"
    Image.new("RGB", (100, 100), "green").save(image_a)
    Image.new("RGB", (100, 100), "yellow").save(image_b)
    (labels / "a.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    (labels / "b.txt").write_text(
        "1 0.5 0.5 0.4 0.4\n3 0.4 0.4 0.2 0.2\n",
        encoding="utf-8",
    )
    (detection / "data.yaml").write_text(
        "nc: 4\nnames: [merged, ripe, old, bad]\n",
        encoding="utf-8",
    )

    manifest = classification / "manifest.csv"
    _write_manifest(
        manifest,
        [
            ["val", str(detection / "images" / "val" / "a.jpg"), "crop_a0.jpg", 0, CLASS_NAMES[0], 0],
            ["val", str(detection / "images" / "val" / "a.jpg"), "crop_a1.jpg", 1, CLASS_NAMES[1], 1],
            ["train", str(image_b), "crop_b0.jpg", 2, CLASS_NAMES[2], 0],
            ["train", str(image_b), "crop_b1.jpg", 0, CLASS_NAMES[0], 1],
        ],
    )

    report, changes = plan_migration(detection, manifest)
    assert report.unresolved_labels == 0
    assert report.mismatched_objects == 0
    assert report.changed_labels == 2
    assert report.new_class_counts == {0: 1, 1: 1, 2: 1, 4: 1}
    assert any("Preserved newer detection label" in issue for issue in report.issues)
    assert len(changes) == 2

    applied = apply_migration(detection, classification, manifest)
    assert applied.backup_path is not None
    assert applied.backup_path.exists()
    assert (labels / "a.txt").read_text(encoding="utf-8").splitlines()[1].startswith("1 ")
    assert (labels / "b.txt").read_text(encoding="utf-8").splitlines()[0].startswith("2 ")
    assert (labels / "b.txt").read_text(encoding="utf-8").splitlines()[1].startswith("4 ")
    assert tuple(names_from_yaml(load_yaml(detection / "data.yaml"))) == CLASS_NAMES


def test_migration_blocks_manifest_mismatch(tmp_path: Path) -> None:
    detection = tmp_path / "detection"
    classification = tmp_path / "classification"
    image_dir = detection / "images" / "train"
    label_dir = detection / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "bad.jpg"
    Image.new("RGB", (20, 20)).save(image)
    (label_dir / "bad.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    manifest = classification / "manifest.csv"
    _write_manifest(manifest, [])

    report, _ = plan_migration(detection, manifest)
    assert report.unresolved_labels == 1
