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
    SimpleClassificationEditor,
    SimpleClassificationError,
    scan_simple_classification_dataset,
)


def _image(path: Path, color) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (31, 23), color).save(path)
    return path


def _assert_editor_matches_model(editor, current) -> None:
    expected = Counter(sample.class_id for sample in current.values())
    assert editor.class_counts == tuple(
        expected[class_id] for class_id in range(len(editor.class_names))
    )
    assert editor.total_images == len(current)
    paths = [sample.image_path for sample in current.values()]
    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize("seed", [20260810, 0xC0FFEE, 0x5EED])
def test_seeded_state_machine_preserves_paths_bytes_and_counts(
    tmp_path: Path,
    seed: int,
) -> None:
    root = tmp_path / "data"
    for class_id in range(6):
        (root / f"class_{class_id:02d}").mkdir(parents=True)
    for item_id in range(18):
        _image(
            root
            / f"class_{item_id % 6:02d}"
            / f"batch_{item_id % 3}"
            / f"item_{item_id:02d}.png",
            (item_id * 11, item_id * 7, item_id * 3),
        )

    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    original = {
        sample.image_path.stem: sample for sample in index.samples
    }
    original_bytes = {
        key: sample.image_path.read_bytes()
        for key, sample in original.items()
    }
    current = dict(original)
    operation_stack = []
    rng = random.Random(seed)

    for _ in range(300):
        existing = sorted(current)
        roll = rng.random()
        if operation_stack and (roll < 0.24 or not existing):
            action, key, old_sample, new_value = operation_stack.pop()
            restored = editor.undo_latest()
            assert restored is not None
            if action == "move":
                assert restored.action == "change_classification"
                assert restored.sample == old_sample
                assert restored.replaced_path == new_value.image_path
            else:
                assert restored.action == "delete_classification"
                assert restored.sample == old_sample
                assert restored.replaced_path == new_value
            current[key] = old_sample
        elif existing and roll < 0.78:
            key = rng.choice(existing)
            old_sample = current[key]
            targets = [
                class_id
                for class_id in range(len(index.class_names))
                if class_id != old_sample.class_id
            ]
            changed = editor.change_class(
                old_sample,
                rng.choice(targets),
            )
            operation_stack.append(("move", key, old_sample, changed))
            current[key] = changed
        elif existing:
            key = rng.choice(existing)
            old_sample = current.pop(key)
            archive = editor.delete_sample(old_sample)
            operation_stack.append(("delete", key, old_sample, archive))
        _assert_editor_matches_model(editor, current)
        for key, sample in current.items():
            assert sample.image_path.read_bytes() == original_bytes[key]

    while operation_stack:
        action, key, old_sample, new_value = operation_stack.pop()
        restored = editor.undo_latest()
        assert restored is not None
        assert restored.sample == old_sample
        if action == "move":
            assert restored.replaced_path == new_value.image_path
        else:
            assert restored.replaced_path == new_value
        current[key] = old_sample
        _assert_editor_matches_model(editor, current)

    assert editor.undo_latest() is None
    assert current == original
    rescanned = scan_simple_classification_dataset(root)
    assert {sample.image_path for sample in rescanned.samples} == {
        sample.image_path for sample in original.values()
    }
    for key, sample in original.items():
        assert sample.image_path.read_bytes() == original_bytes[key]


def test_collision_storm_never_overwrites_and_undo_is_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source_paths = []
    for class_id in range(20):
        source_paths.append(
            _image(
                root
                / f"source_{class_id:02d}"
                / "nested"
                / "same.name.JPG",
                (class_id * 10, class_id * 5, class_id * 3),
            )
        )
    target_original = _image(
        root / "zz_target" / "nested" / "same.name.JPG",
        (255, 255, 255),
    )
    expected_bytes = {
        path: path.read_bytes() for path in (*source_paths, target_original)
    }
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    target_id = index.class_names.index("zz_target")

    moved = [
        editor.change_class(sample, target_id)
        for sample in index.samples
        if sample.class_id != target_id
    ]

    target_files = sorted((root / "zz_target" / "nested").glob("*.JPG"))
    assert len(target_files) == 21
    assert len({path.name.casefold() for path in target_files}) == 21
    assert target_original.read_bytes() == expected_bytes[target_original]
    assert {sample.image_path for sample in moved} == set(target_files) - {
        target_original
    }
    assert editor.class_counts[target_id] == 21

    for expected_path in reversed(source_paths):
        restored = editor.undo_latest()
        assert restored is not None
        assert restored.sample.image_path == expected_path
        assert expected_path.read_bytes() == expected_bytes[expected_path]

    assert editor.undo_latest() is None
    assert editor.class_counts == (1,) * 21
    assert target_original.read_bytes() == expected_bytes[target_original]


def test_reopened_editor_undoes_persisted_operations_with_corrupt_tail(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    first = _image(root / "a" / "first.png", (10, 20, 30))
    second = _image(root / "b" / "second.png", (40, 50, 60))
    (root / "c").mkdir(parents=True)
    original_bytes = {first: first.read_bytes(), second: second.read_bytes()}
    first_index = scan_simple_classification_dataset(root)
    first_editor = SimpleClassificationEditor(first_index)
    changed = first_editor.change_class(first_index.samples[0], 1)
    archived = first_editor.delete_sample(
        next(sample for sample in first_index.samples if sample.image_path == second)
    )
    with first_editor.journal.path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated-json\n")
        handle.write("not-json\n")

    reopened_index = scan_simple_classification_dataset(root)
    reopened = SimpleClassificationEditor(reopened_index)
    undo_delete = reopened.undo_latest()
    undo_change = reopened.undo_latest()

    assert undo_delete is not None
    assert undo_delete.sample.image_path == second
    assert undo_change is not None
    assert undo_change.sample.image_path == first
    assert reopened.undo_latest() is None
    assert not archived.exists()
    assert not changed.image_path.exists()
    assert first.read_bytes() == original_bytes[first]
    assert second.read_bytes() == original_bytes[second]
    assert reopened.class_counts == (1, 1, 0)


def test_concurrent_move_and_delete_same_sample_allow_exactly_one_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png", (1, 2, 3))
    (root / "b").mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    sample = index.samples[0]
    barrier = threading.Barrier(2)

    def run(action):
        barrier.wait()
        try:
            if action == "move":
                return action, editor.change_class(sample, 1), None
            return action, editor.delete_sample(sample), None
        except Exception as error:
            return action, None, error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("move", "delete")))

    successes = [result for result in results if result[2] is None]
    failures = [result for result in results if result[2] is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0][2], SimpleClassificationError)
    committed = [
        record
        for record in editor.journal.records()
        if record.get("status") == "committed"
    ]
    assert len(committed) == 1
    assert editor.total_images in (0, 1)
    assert sum(editor.class_counts) == editor.total_images
    assert not source.exists()
    assert len(list(root.rglob("*.png"))) == editor.total_images


@pytest.mark.parametrize("operation", ["move", "delete"])
def test_filesystem_failure_before_mutation_keeps_state_exact(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.png", (1, 2, 3))
    (root / "b").mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    sample = index.samples[0]
    original_bytes = source.read_bytes()

    if operation == "move":
        original_replace = Path.replace

        def fail_replace(path, target):
            if path == source:
                raise PermissionError("image locked")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_replace)
        call = lambda: editor.change_class(sample, 1)
    else:
        monkeypatch.setattr(
            simple_editor_module.shutil,
            "move",
            lambda source_path, target_path: (_ for _ in ()).throw(
                PermissionError("archive locked")
            ),
        )
        call = lambda: editor.delete_sample(sample)

    with pytest.raises(PermissionError, match="locked"):
        call()

    assert source.read_bytes() == original_bytes
    assert editor.class_counts == (1, 0)
    assert editor.journal.records() == []


def test_target_nested_reparse_introduced_after_scan_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "nested" / "one.png", (1, 2, 3))
    target_nested = root / "b" / "nested"
    target_nested.mkdir(parents=True)
    index = scan_simple_classification_dataset(root)
    editor = SimpleClassificationEditor(index)
    original_reparse_check = simple_editor_module._is_reparse_point
    monkeypatch.setattr(
        simple_editor_module,
        "_is_reparse_point",
        lambda path: (
            Path(path) == target_nested
            or original_reparse_check(Path(path))
        ),
    )

    with pytest.raises(SimpleClassificationError, match="symlink/junction"):
        editor.change_class(index.samples[0], 1)

    assert source.is_file()
    assert editor.class_counts == (1, 0)
    assert editor.journal.records() == []


def test_hidden_and_reserved_directories_never_become_classes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    visible = _image(root / "visible" / "one.png", (1, 2, 3))
    _image(root / ".hidden" / "hidden.png", (4, 5, 6))
    _image(root / ARCHIVE_DIR_NAME / "fake.png", (7, 8, 9))
    (root / "data.yaml").write_text("note: ordinary-file", encoding="utf-8")
    (root / "README.txt").write_text("ignored", encoding="utf-8")

    index = scan_simple_classification_dataset(root)

    assert index.class_names == ("visible",)
    assert [sample.image_path for sample in index.samples] == [visible.resolve()]
