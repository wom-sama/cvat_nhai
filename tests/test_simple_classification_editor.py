from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

import cvat_nhai.simple_classification_editor as simple_editor_module
from cvat_nhai.models import ClassificationSample
from cvat_nhai.simple_classification_editor import (
    SimpleClassificationEditor,
    SimpleClassificationError,
    scan_simple_classification_dataset,
)


def _image(path: Path, color: str = "green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)
    return path


def _dataset(root: Path) -> Path:
    _image(root / "z_bad" / "nested" / "bad.JPG", "red")
    _image(root / "a_green" / "green.png", "green")
    (root / "middle_empty").mkdir(parents=True)
    (root / "a_green" / "notes.txt").write_text(
        "not an image",
        encoding="utf-8",
    )
    return root


def _sample(index, filename: str) -> ClassificationSample:
    return next(
        sample for sample in index.samples if sample.image_path.name == filename
    )


def test_scan_discovers_dynamic_classes_counts_and_nested_images(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "data")
    (root / ".ignored_cache").mkdir()

    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)

    assert index.root == root.resolve()
    assert index.data_yaml is None
    assert index.class_names == ("a_green", "middle_empty", "z_bad")
    assert [sample.image_path.name for sample in index.samples] == [
        "green.png",
        "bad.JPG",
    ]
    assert all(sample.split == "data" for sample in index.samples)
    assert editor.class_counts == (1, 0, 1)
    assert editor.total_images == 2


def test_scan_accepts_unicode_spaces_many_classes_and_empty_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    names = [
        "Class 10",
        "Class_2",
        "Xoai chin",
        "Đặc biệt",
        "alpha",
        "beta",
        "gamma",
        "omega",
    ]
    for name in names:
        (root / name).mkdir(parents=True)

    index = scan_simple_classification_dataset(root)

    assert len(index.class_names) == 8
    assert set(index.class_names) == set(names)
    assert index.samples == ()
    assert SimpleClassificationEditor(index).class_counts == (0,) * 8


@pytest.mark.parametrize(
    "layout, message",
    [
        ("missing", "khong ton tai"),
        ("no_classes", "Khong tim thay"),
        ("root_image", "truc tiep"),
        ("structured", "YOLO/class_f"),
    ],
)
def test_scan_rejects_ambiguous_or_invalid_roots(
    tmp_path: Path,
    layout: str,
    message: str,
) -> None:
    root = tmp_path / "data"
    if layout != "missing":
        root.mkdir()
    if layout == "root_image":
        _image(root / "orphan.jpg")
    elif layout == "structured":
        (root / "train").mkdir()
        (root / "data.yaml").write_text("names: [a]", encoding="utf-8")

    with pytest.raises(SimpleClassificationError, match=message):
        scan_simple_classification_dataset(root)


def test_scan_rejects_reparse_points_without_following_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _dataset(tmp_path / "data")
    nested = root / "a_green" / "external_link"
    nested.mkdir()
    original = simple_editor_module._is_reparse_point

    monkeypatch.setattr(
        simple_editor_module,
        "_is_reparse_point",
        lambda path: Path(path) == nested or original(Path(path)),
    )

    with pytest.raises(SimpleClassificationError, match="symlink/junction"):
        scan_simple_classification_dataset(root)


def test_scan_rejects_selected_root_reparse_point(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _dataset(tmp_path / "data")
    monkeypatch.setattr(
        simple_editor_module,
        "_is_reparse_point",
        lambda path: Path(path) == root,
    )

    with pytest.raises(SimpleClassificationError, match="symlink/junction"):
        scan_simple_classification_dataset(root)


def test_change_class_moves_file_preserves_subpath_and_updates_counts(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    sample = _sample(index, "bad.JPG")
    target_id = index.class_names.index("a_green")

    changed = editor.change_class(sample, target_id)

    assert not sample.image_path.exists()
    assert changed.image_path == root / "a_green" / "nested" / "bad.JPG"
    assert changed.image_path.is_file()
    assert changed.class_name == "a_green"
    assert editor.class_counts == (2, 0, 0)
    assert editor.total_images == 2


def test_change_to_current_class_is_noop_without_journal_record(
    tmp_path: Path,
) -> None:
    index = scan_simple_classification_dataset(_dataset(tmp_path / "data"))
    editor = SimpleClassificationEditor(index)
    sample = _sample(index, "green.png")

    assert editor.change_class(sample, sample.class_id) is sample
    assert editor.journal.records() == []
    assert sample.image_path.exists()


def test_change_class_never_overwrites_name_collision(tmp_path: Path) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "nested" / "same.jpg", "green")
    existing = _image(root / "b" / "nested" / "same.jpg", "red")
    existing_bytes = existing.read_bytes()
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)

    changed = editor.change_class(_sample(index, source.name), 1)

    assert changed.image_path.name == "same__classedit001.jpg"
    assert changed.image_path.is_file()
    assert existing.read_bytes() == existing_bytes
    assert editor.class_counts == (0, 2)


def test_delete_archives_outside_dataset_and_rescan_stays_clean(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    sample = _sample(index, "green.png")

    archived = editor.delete_sample(sample)

    assert not sample.image_path.exists()
    assert archived.is_file()
    assert root.resolve() not in archived.resolve().parents
    assert editor.class_counts == (0, 0, 1)
    rescanned = scan_simple_classification_dataset(root)
    assert len(rescanned.samples) == 1
    assert set(rescanned.class_names) == {"a_green", "middle_empty", "z_bad"}


def test_change_delete_and_undo_are_lifo_and_byte_exact(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    original = _sample(index, "green.png")
    original_bytes = original.image_path.read_bytes()
    class_bad = index.class_names.index("z_bad")

    changed = editor.change_class(original, class_bad)
    archived = editor.delete_sample(changed)
    restored_delete = editor.undo_latest()
    restored_change = editor.undo_latest()

    assert restored_delete is not None
    assert restored_delete.action == "delete_classification"
    assert restored_delete.sample == changed
    assert restored_change is not None
    assert restored_change.action == "change_classification"
    assert restored_change.sample == original
    assert original.image_path.read_bytes() == original_bytes
    assert not changed.image_path.exists()
    assert not archived.exists()
    assert editor.class_counts == (1, 0, 1)
    assert editor.undo_latest() is None


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_operation_rolls_back_when_journal_write_fails(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    sample = _sample(index, "green.png")
    initial_counts = editor.class_counts
    monkeypatch.setattr(
        editor.journal,
        "append",
        lambda payload: (_ for _ in ()).throw(OSError("journal locked")),
    )

    with pytest.raises(OSError, match="journal locked"):
        if operation == "change":
            editor.change_class(sample, index.class_names.index("z_bad"))
        else:
            editor.delete_sample(sample)

    assert sample.image_path.is_file()
    assert editor.class_counts == initial_counts
    matches = [
        path.resolve()
        for path in root.parent.rglob("green.png")
        if path.is_file()
    ]
    assert matches == [sample.image_path.resolve()]


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_undo_rolls_back_when_undo_journal_write_fails(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    original = _sample(index, "green.png")
    if operation == "change":
        current = editor.change_class(
            original,
            index.class_names.index("z_bad"),
        )
    else:
        archived = editor.delete_sample(original)
        current = None
    expected_counts = editor.class_counts
    monkeypatch.setattr(
        editor.journal,
        "append",
        lambda payload: (_ for _ in ()).throw(OSError("undo journal locked")),
    )

    with pytest.raises(OSError, match="undo journal locked"):
        editor.undo_latest()

    assert editor.class_counts == expected_counts
    if operation == "change":
        assert current is not None and current.image_path.is_file()
        assert not original.image_path.exists()
    else:
        assert archived.is_file()
        assert not original.image_path.exists()


def test_rejects_invalid_or_external_sample_without_mutation(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    outside = _image(tmp_path / "outside.jpg")
    forged = ClassificationSample(
        image_path=outside,
        split="data",
        class_id=0,
        class_name=index.class_names[0],
    )

    with pytest.raises(SimpleClassificationError, match="ngoai class"):
        editor.change_class(forged, 2)
    with pytest.raises(SimpleClassificationError, match="Class 99"):
        editor.change_class(index.samples[0], 99)
    text_sample = ClassificationSample(
        image_path=root / "a_green" / "notes.txt",
        split="data",
        class_id=0,
        class_name=index.class_names[0],
    )
    with pytest.raises(SimpleClassificationError, match="dinh dang anh"):
        editor.change_class(text_sample, 2)

    assert outside.is_file()
    assert editor.class_counts == (1, 0, 1)


@pytest.mark.parametrize("operation", ["change", "delete"])
def test_undo_refuses_to_overwrite_external_file(
    tmp_path: Path,
    operation: str,
) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    original = _sample(index, "green.png")
    if operation == "change":
        current = editor.change_class(
            original,
            index.class_names.index("z_bad"),
        )
    else:
        archived = editor.delete_sample(original)
        current = None
    collision = _image(original.image_path, "blue")
    collision_bytes = collision.read_bytes()

    with pytest.raises(SimpleClassificationError, match="da ton tai"):
        editor.undo_latest()

    assert collision.read_bytes() == collision_bytes
    if operation == "change":
        assert current is not None and current.image_path.is_file()
    else:
        assert archived.is_file()


def test_undo_refuses_to_recreate_class_removed_externally(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "only.jpg")
    (root / "b").mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    changed = editor.change_class(_sample(index, source.name), 1)
    (root / "a").rmdir()

    with pytest.raises(SimpleClassificationError, match="doi/xoa"):
        editor.undo_latest()

    assert changed.image_path.is_file()
    assert not (root / "a").exists()


def test_archive_must_be_outside_dataset(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    index = scan_simple_classification_dataset(root)

    with pytest.raises(SimpleClassificationError, match="ngoai"):
        SimpleClassificationEditor(index, archive_root=root / ".archive")


def test_concurrent_changes_are_serialized_and_counts_remain_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    for index in range(24):
        _image(root / "source" / "batch" / f"image_{index:02d}.jpg")
    (root / "target").mkdir(parents=True)
    dataset = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(dataset)
    target_id = dataset.class_names.index("target")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda sample: editor.change_class(sample, target_id),
                dataset.samples,
            )
        )

    assert len(results) == 24
    assert all(result.image_path.is_file() for result in results)
    assert len({result.image_path for result in results}) == 24
    assert editor.class_counts == (0, 24)
    assert editor.total_images == 24
