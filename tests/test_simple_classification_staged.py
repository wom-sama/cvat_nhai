import hashlib
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

import cvat_nhai.simple_classification_editor as simple_editor_module
from cvat_nhai.simple_classification_editor import (
    ARCHIVE_DIR_NAME,
    SimpleClassificationError,
    StagedSimpleClassificationEditor,
    scan_simple_classification_dataset,
)


def _image(path: Path, color=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (37, 29), color).save(path)
    return path


def _source_snapshot(root: Path):
    return {
        path.relative_to(root): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sample(index, name: str):
    return next(
        sample for sample in index.samples if sample.image_path.name == name
    )


def _editor(root: Path) -> StagedSimpleClassificationEditor:
    return StagedSimpleClassificationEditor(
        scan_simple_classification_dataset(root)
    )


def test_stage_change_delete_and_undo_never_mutate_source(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first = _image(root / "a" / "nested" / "first.png", (1, 2, 3))
    second = _image(root / "b" / "second.jpg", (4, 5, 6))
    (root / "c").mkdir(parents=True)
    before = _source_snapshot(root)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    assert not (root.parent / ARCHIVE_DIR_NAME).exists()

    changed = editor.change_class(_sample(index, "first.png"), 2)
    deleted_path = editor.delete_sample(_sample(index, "second.jpg"))

    assert changed.image_path == first
    assert changed.class_id == 2
    assert deleted_path == second
    assert first.is_file() and second.is_file()
    assert editor.class_counts == (0, 0, 1)
    assert editor.staged_change_count == 2
    assert editor.changed_count == 1
    assert editor.deleted_count == 1
    assert editor.has_unexported_changes
    assert _source_snapshot(root) == before
    assert not (root.parent / ARCHIVE_DIR_NAME).exists()

    restored_delete = editor.undo_latest()
    restored_change = editor.undo_latest()

    assert restored_delete is not None
    assert restored_delete.action == "delete_classification"
    assert restored_delete.sample.image_path == second
    assert restored_change is not None
    assert restored_change.action == "change_classification"
    assert restored_change.sample.image_path == first
    assert editor.class_counts == (1, 1, 0)
    assert editor.staged_change_count == 0
    assert not editor.has_unexported_changes
    assert editor.undo_latest() is None
    assert _source_snapshot(root) == before


def test_export_publishes_complete_copy_and_marks_exact_revision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    first = _image(root / "a" / "nested" / "first.png", (1, 2, 3))
    second = _image(root / "b" / "second.jpg", (4, 5, 6))
    (root / "c").mkdir(parents=True)
    before = _source_snapshot(root)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    editor.change_class(_sample(index, "first.png"), 2)
    editor.delete_sample(_sample(index, "second.jpg"))
    events = []
    destination = tmp_path / "published" / "data_edited"

    report = editor.export_dataset(destination, progress_callback=events.append)

    assert report.destination == destination.resolve()
    assert report.images == 1
    assert report.changed == 1
    assert report.deleted == 1
    assert report.class_counts == (0, 0, 1)
    assert (destination / "c" / "nested" / "first.png").read_bytes() == (
        first.read_bytes()
    )
    assert not list((destination / "a").rglob("*.*"))
    assert not list((destination / "b").rglob("*.*"))
    assert not (destination / "data.yaml").exists()
    assert not (destination / "manifest.csv").exists()
    assert events[0]["value"] == 0
    assert events[-1]["value"] == events[-1]["maximum"]
    assert _source_snapshot(root) == before

    editor.mark_exported(report.revision)
    assert not editor.has_unexported_changes
    editor.undo_latest()
    assert editor.has_unexported_changes
    assert _source_snapshot(root) == before


def test_export_resolves_file_and_directory_name_collisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    paths = [
        _image(root / "a" / "nested" / "same.JPG", (1, 0, 0)),
        _image(root / "b" / "nested" / "same.JPG", (2, 0, 0)),
        _image(root / "c" / "nested" / "same.JPG", (3, 0, 0)),
        _image(root / "d" / "node.jpg", (4, 0, 0)),
        _image(root / "e" / "node.jpg" / "child.png", (5, 0, 0)),
    ]
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    target_id = index.class_names.index("c")
    for sample in index.samples:
        if sample.class_id != target_id:
            editor.change_class(sample, target_id)

    destination = tmp_path / "output"
    report = editor.export_dataset(destination)
    output_files = sorted(
        path for path in (destination / "c").rglob("*") if path.is_file()
    )

    assert report.images == len(paths)
    assert len(output_files) == len(paths)
    assert len({str(path).casefold() for path in output_files}) == len(paths)
    assert {path.read_bytes() for path in output_files} == {
        path.read_bytes() for path in paths
    }


@pytest.mark.parametrize("destination_kind", ["same", "inside", "ancestor"])
def test_export_rejects_any_source_destination_overlap(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    root = tmp_path / "parent" / "data"
    _image(root / "a" / "one.png")
    editor = _editor(root)
    destination = {
        "same": root,
        "inside": root / "new_data",
        "ancestor": root.parent,
    }[destination_kind]

    with pytest.raises(SimpleClassificationError, match="ngoai dataset nguon"):
        editor.export_dataset(destination)

    assert (root / "a" / "one.png").is_file()


def test_export_rejects_nonempty_destination_without_touching_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png")
    destination = tmp_path / "output"
    marker = destination / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(SimpleClassificationError, match="phai rong"):
        _editor(root).export_dataset(destination)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert source.is_file()


def test_cancel_after_copy_removes_staging_and_keeps_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    for index in range(4):
        _image(root / "a" / f"image_{index}.png", (index, 2, 3))
    before = _source_snapshot(root)
    destination = tmp_path / "output"
    cancel_event = threading.Event()

    def cancel_after_first(payload):
        if payload["value"] >= 1:
            cancel_event.set()

    with pytest.raises(SimpleClassificationError, match="huy an toan"):
        _editor(root).export_dataset(
            destination,
            progress_callback=cancel_after_first,
            cancel_event=cancel_event,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".cvat_nhai_simple_export_*"))
    assert _source_snapshot(root) == before


def test_copy_failure_restores_preexisting_empty_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    for index in range(3):
        _image(root / "a" / f"image_{index}.png", (index, 2, 3))
    destination = tmp_path / "output"
    destination.mkdir()
    before = _source_snapshot(root)
    original_copy = simple_editor_module.shutil.copy2
    calls = []

    def fail_second(source, target):
        calls.append(source)
        if len(calls) == 2:
            raise PermissionError("destination locked")
        return original_copy(source, target)

    monkeypatch.setattr(simple_editor_module.shutil, "copy2", fail_second)

    with pytest.raises(PermissionError, match="locked"):
        _editor(root).export_dataset(destination)

    assert destination.is_dir() and not any(destination.iterdir())
    assert not list(tmp_path.glob(".cvat_nhai_simple_export_*"))
    assert _source_snapshot(root) == before


@pytest.mark.parametrize("preexisting", [False, True])
def test_atomic_publish_failure_never_leaves_partial_dataset(
    tmp_path: Path,
    monkeypatch,
    preexisting: bool,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png")
    destination = tmp_path / "output"
    if preexisting:
        destination.mkdir()
    monkeypatch.setattr(
        simple_editor_module.os,
        "replace",
        lambda source_path, target_path: (_ for _ in ()).throw(
            OSError("atomic publish blocked")
        ),
    )

    with pytest.raises(OSError, match="publish blocked"):
        _editor(root).export_dataset(destination)

    if preexisting:
        assert destination.is_dir() and not any(destination.iterdir())
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".cvat_nhai_simple_export_*"))
    assert source.is_file()


def test_destination_created_during_export_is_preserved_and_publish_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    _image(root / "a" / "one.png")
    destination = tmp_path / "output"
    marker = destination / "external.txt"

    def create_collision(payload):
        if payload["value"] == 1 and not destination.exists():
            destination.mkdir()
            marker.write_text("external", encoding="utf-8")

    with pytest.raises(SimpleClassificationError, match="da xuat hien"):
        _editor(root).export_dataset(
            destination,
            progress_callback=create_collision,
        )

    assert marker.read_text(encoding="utf-8") == "external"
    assert not list(tmp_path.glob(".cvat_nhai_simple_export_*"))


def test_corrupt_supported_image_is_copied_byte_exact(tmp_path: Path) -> None:
    root = tmp_path / "data"
    corrupt = root / "a" / "corrupt.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not-an-image-but-preserve-it")

    report = _editor(root).export_dataset(tmp_path / "output")

    assert report.images == 1
    assert (tmp_path / "output" / "a" / "corrupt.jpg").read_bytes() == (
        corrupt.read_bytes()
    )


def test_mark_exported_rejects_stale_snapshot_revision(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root / "a" / "one.png")
    (root / "b").mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    editor.change_class(index.samples[0], 1)
    report = editor.export_dataset(tmp_path / "output")
    editor.undo_latest()

    with pytest.raises(SimpleClassificationError, match="thay doi"):
        editor.mark_exported(report.revision)

    assert editor.has_unexported_changes is False
    assert (root / "a" / "one.png").is_file()


def test_concurrent_move_and_delete_commit_only_one_staged_operation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png")
    (root / "b").mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    sample = index.samples[0]
    barrier = threading.Barrier(2)

    def run(action):
        barrier.wait()
        try:
            if action == "move":
                return editor.change_class(sample, 1), None
            return editor.delete_sample(sample), None
        except Exception as error:
            return None, error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("move", "delete")))

    successes = [result for result, error in results if error is None]
    failures = [error for result, error in results if error is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], SimpleClassificationError)
    assert editor.staged_change_count == 1
    assert sum(editor.class_counts) == editor.total_images
    assert source.is_file()


def test_missing_source_aborts_export_without_partial_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png")
    editor = _editor(root)
    source.unlink()

    with pytest.raises(SimpleClassificationError, match="khong con ton tai"):
        editor.export_dataset(tmp_path / "output")

    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".cvat_nhai_simple_export_*"))


@pytest.mark.parametrize("seed", [20260811, 0xDADA, 0xBADC0DE])
def test_seeded_staging_state_machine_and_periodic_exports_are_exact(
    tmp_path: Path,
    seed: int,
) -> None:
    root = tmp_path / "data"
    for class_id in range(7):
        (root / f"class_{class_id:02d}").mkdir(parents=True)
    for image_id in range(35):
        _image(
            root
            / f"class_{image_id % 7:02d}"
            / f"batch_{image_id % 3}"
            / f"item_{image_id:03d}.png",
            (
                (image_id * 7) % 255,
                (image_id * 11) % 255,
                (image_id * 13) % 255,
            ),
        )
    before = _source_snapshot(root)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    original = {
        sample.image_path: sample for sample in index.samples
    }
    current = dict(original)
    history = []
    rng = random.Random(seed)

    for step in range(420):
        existing = sorted(current, key=lambda path: str(path).casefold())
        roll = rng.random()
        if history and (roll < 0.22 or not existing):
            action, path, old_sample, new_sample = history.pop()
            restored = editor.undo_latest()
            assert restored is not None
            assert restored.sample == old_sample
            if action == "change":
                assert restored.action == "change_classification"
                assert new_sample is not None
            else:
                assert restored.action == "delete_classification"
            current[path] = old_sample
        elif existing and roll < 0.76:
            path = rng.choice(existing)
            old_sample = current[path]
            target_ids = [
                class_id
                for class_id in range(len(index.class_names))
                if class_id != old_sample.class_id
            ]
            changed = editor.change_class(
                old_sample,
                rng.choice(target_ids),
            )
            history.append(("change", path, old_sample, changed))
            current[path] = changed
        elif existing:
            path = rng.choice(existing)
            old_sample = current.pop(path)
            editor.delete_sample(old_sample)
            history.append(("delete", path, old_sample, None))

        expected_counts = Counter(
            sample.class_id for sample in current.values()
        )
        assert editor.class_counts == tuple(
            expected_counts[class_id]
            for class_id in range(len(index.class_names))
        )
        assert editor.total_images == len(current)
        assert _source_snapshot(root) == before

        if step % 105 == 104:
            destination = tmp_path / f"export_{seed}_{step}"
            report = editor.export_dataset(destination)
            exported = scan_simple_classification_dataset(destination)
            assert len(exported.samples) == len(current)
            assert Counter(sample.class_id for sample in exported.samples) == (
                expected_counts
            )
            expected_hashes = Counter(
                hashlib.sha256(path.read_bytes()).digest()
                for path in current
            )
            actual_hashes = Counter(
                hashlib.sha256(sample.image_path.read_bytes()).digest()
                for sample in exported.samples
            )
            assert actual_hashes == expected_hashes
            editor.mark_exported(report.revision)
            assert not editor.has_unexported_changes


def test_concurrent_edit_during_export_cannot_mark_stale_copy_current(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    first = _image(root / "a" / "first.png", (1, 2, 3))
    second = _image(root / "a" / "second.png", (4, 5, 6))
    (root / "b").mkdir(parents=True)
    before = _source_snapshot(root)
    index = scan_simple_classification_dataset(root)
    editor = StagedSimpleClassificationEditor(index)
    editor.change_class(_sample(index, "first.png"), 1)
    original_copy = simple_editor_module.shutil.copy2
    copy_started = threading.Event()
    release_copy = threading.Event()
    copy_calls = []

    def blocking_copy(source, target):
        copy_calls.append(source)
        if len(copy_calls) == 1:
            copy_started.set()
            assert release_copy.wait(timeout=5)
        return original_copy(source, target)

    monkeypatch.setattr(simple_editor_module.shutil, "copy2", blocking_copy)
    destination = tmp_path / "output"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(editor.export_dataset, destination)
        assert copy_started.wait(timeout=5)
        editor.change_class(_sample(index, "second.png"), 1)
        release_copy.set()
        report = future.result(timeout=10)

    with pytest.raises(SimpleClassificationError, match="thay doi"):
        editor.mark_exported(report.revision)

    exported = scan_simple_classification_dataset(destination)
    exported_classes_by_bytes = {
        sample.image_path.read_bytes(): sample.class_name
        for sample in exported.samples
    }
    assert exported_classes_by_bytes[first.read_bytes()] == "b"
    assert exported_classes_by_bytes[second.read_bytes()] == "a"
    assert editor.class_counts == (0, 2)
    assert editor.has_unexported_changes
    assert _source_snapshot(root) == before


def test_export_rejects_destination_reparse_detected_after_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    _image(root / "a" / "one.png")
    destination_parent = tmp_path / "exports"
    destination_parent.mkdir()
    original_check = simple_editor_module._is_reparse_point
    monkeypatch.setattr(
        simple_editor_module,
        "_is_reparse_point",
        lambda path: (
            Path(path) == destination_parent
            or original_check(Path(path))
        ),
    )

    with pytest.raises(SimpleClassificationError, match="symlink/junction"):
        _editor(root).export_dataset(destination_parent / "output")

    assert not (destination_parent / "output").exists()
