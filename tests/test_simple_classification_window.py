import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt

import cvat_nhai.main_window as main_window_module
from cvat_nhai.constants import CLASS_NAMES
from cvat_nhai.main_window import MainWindow


def _image(path: Path, color: str = "green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 72), color).save(path)
    return path


def _open_simple_mode(window: MainWindow, root: Path) -> None:
    mode_index = window.mode_combo.findData("simple_class")
    assert mode_index >= 0
    window.mode_combo.setCurrentIndex(mode_index)
    window.source_edit.setText(str(root))
    window.scan_source()


def test_simple_mode_dynamic_classes_move_filter_delete_and_undo(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    first = _image(root / "class_00" / "a.jpg")
    second = _image(root / "class_00" / "b.jpg")
    for class_id in range(1, 7):
        _image(
            root / f"class_{class_id:02d}" / f"image_{class_id}.jpg",
            "orange",
        )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: (
            window.current_path == first
            and not window.active_tasks
            and not window.canvas.is_loading
        ),
        timeout=5000,
    )

    assert window.active_class_names == tuple(
        f"class_{index:02d}" for index in range(7)
    )
    assert len(window.class_buttons) == 7
    assert all(button.isVisible() for button in window.class_buttons)
    assert "2 anh" in window.class_buttons[0].text()
    assert "1 anh" in window.class_buttons[6].text()
    assert window.queue_label.text() == "8 anh data don gian | 7 class"
    assert not window.editor_split_filter_combo.isVisible()
    assert window.editor_class_filter_combo.isVisible()
    assert not window.canvas._annotation_enabled
    assert not (root / "data.yaml").exists()
    assert not (root / "manifest.csv").exists()

    qtbot.keyClick(window, Qt.Key_7)
    assert window.selected_class == 6
    assert window._classification_is_dirty()
    qtbot.keyClick(window, Qt.Key_Z, modifier=Qt.ControlModifier)
    assert window.selected_class == 0
    assert not window._classification_is_dirty()

    window.select_class(6)
    window.commit_current()
    moved = root / "class_06" / "a.jpg"
    qtbot.waitUntil(
        lambda: (
            moved.is_file()
            and not first.exists()
            and window.current_path == second
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert "1 anh" in window.class_buttons[0].text()
    assert "2 anh" in window.class_buttons[6].text()
    assert window.classification_samples_by_path[moved].class_id == 6
    assert window.selected_class == 0
    assert not window._classification_is_dirty()

    window.editor_class_filter_combo.setCurrentIndex(
        window.editor_class_filter_combo.findData(6)
    )
    qtbot.waitUntil(
        lambda: window.current_path == moved and len(window.images) == 2,
        timeout=1000,
    )
    assert "class 7 (class_06)" in window.queue_label.text()

    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            not moved.exists()
            and moved not in window.classification_samples_by_path
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert "1 anh" in window.class_buttons[6].text()
    assert not any(root.rglob(".cvat_nhai_simple_archive"))
    archive_parent = root.parent / ".cvat_nhai_simple_archive"
    assert list(archive_parent.rglob("a.jpg"))

    qtbot.keyClick(window, Qt.Key_Z, modifier=Qt.ControlModifier)
    qtbot.waitUntil(
        lambda: (
            moved.is_file()
            and moved in window.classification_samples_by_path
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert "2 anh" in window.class_buttons[6].text()
    assert window.current_path == moved


def test_simple_mode_opens_empty_classes_without_fixed_five_class_schema(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "empty_data"
    names = tuple(f"class_{index:02d}" for index in range(12))
    for name in names:
        (root / name).mkdir(parents=True)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)
    qtbot.waitUntil(lambda: not window.active_tasks, timeout=5000)

    assert window.active_class_names == names
    assert len(window.class_buttons) == 12
    assert all("0 anh" in button.text() for button in window.class_buttons)
    assert window.class_buttons[8].toolTip().startswith("Phim 9\n")
    assert window.class_buttons[9].toolTip().startswith(
        "Click de chon class\n"
    )
    window.select_class(11)
    assert window.selected_class == 11
    assert window.current_path is None
    assert window.position_label.text() == "0 / 0"
    assert window.queue_label.text() == "0 anh data don gian | 12 class"
    assert window.schema_badge.text() == "12 class - data don gian"


def test_simple_scan_locks_reentrant_ui_and_shows_delayed_feedback(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    _image(root / "a" / "one.jpg")
    original_scan = main_window_module.scan_simple_classification_dataset
    calls = []

    def slow_scan(path):
        calls.append(path)
        time.sleep(0.3)
        return original_scan(path)

    monkeypatch.setattr(
        main_window_module,
        "scan_simple_classification_dataset",
        slow_scan,
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)

    assert window.busy
    assert not window.mode_combo.isEnabled()
    assert not window.source_edit.isEnabled()
    window.scan_source()
    qtbot.waitUntil(window.busy_overlay.isVisible, timeout=1000)
    qtbot.waitUntil(
        lambda: (
            len(window.images) == 1
            and not window.active_tasks
            and not window.busy
        ),
        timeout=5000,
    )

    assert len(calls) == 1
    assert window.mode_combo.isEnabled()
    assert window.source_edit.isEnabled()
    assert not window.busy_overlay.isVisible()


def test_simple_mode_blocks_reentrant_actions_while_move_is_running(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    source = _image(root / "a" / "one.jpg")
    (root / "b").mkdir(parents=True)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: window.current_path == source and not window.active_tasks,
        timeout=5000,
    )
    original_change = window.classification_manager.change_class

    def slow_change(sample, class_id):
        time.sleep(0.25)
        return original_change(sample, class_id)

    monkeypatch.setattr(
        window.classification_manager,
        "change_class",
        slow_change,
    )
    window.select_class(1)
    window.commit_current()
    window.commit_current()
    window.delete_current()

    assert window.busy
    assert not window.commit_button.isEnabled()
    qtbot.waitUntil(
        lambda: (
            (root / "b" / "one.jpg").is_file()
            and not window.active_tasks
            and not window.busy
        ),
        timeout=5000,
    )
    assert not source.exists()
    assert len(window.classification_manager.journal.records()) == 1
    assert window.classification_manager.total_images == 1


def test_simple_mode_reports_invalid_root_without_freezing_ui(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "invalid"
    _image(root / "orphan.jpg")
    errors = []

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    monkeypatch.setattr(window, "_show_error", errors.append)
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: bool(errors) and not window.active_tasks,
        timeout=5000,
    )

    assert "truc tiep" in errors[0]
    assert window.scan_button.isEnabled()
    assert window.scan_button.text() == "Mo data don gian"
    assert not window.busy


def test_simple_filter_move_delete_to_empty_seek_guard_and_undo(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    class_a = [
        _image(root / "a" / f"a{index}.png", "green")
        for index in range(3)
    ]
    for index in range(2):
        _image(root / "b" / f"b{index}.png", "orange")
    _image(root / "c" / "c0.png", "red")

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: (
            window.current_path == class_a[0]
            and not window.active_tasks
            and not window.canvas.is_loading
        ),
        timeout=5000,
    )
    window.editor_class_filter_combo.setCurrentIndex(
        window.editor_class_filter_combo.findData(0)
    )
    assert window.images == class_a

    window.seek_to_index(2)
    assert window.current_path == class_a[2]
    window.select_class(1)
    assert window._classification_is_dirty()
    window.seek_to_index(0)
    assert window.current_path == class_a[2]
    assert window.progress.value() == 2
    window.reset_annotation()
    window.seek_to_index(0)
    assert window.current_path == class_a[0]

    window.select_class(1)
    window.commit_current()
    moved = root / "b" / "a0.png"
    qtbot.waitUntil(
        lambda: (
            moved.is_file()
            and window.current_path == class_a[1]
            and len(window.images) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.classification_manager.class_counts == (2, 3, 1)

    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            not class_a[1].exists()
            and window.current_path == class_a[2]
            and len(window.images) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            window.current_path is None
            and not window.images
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert "0 anh class 1 (a)" in window.queue_label.text()
    assert window.classification_manager.class_counts == (0, 3, 1)

    window.undo_latest()
    qtbot.waitUntil(
        lambda: (
            class_a[2].is_file()
            and window.current_path == class_a[2]
            and window.images == [class_a[2]]
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.classification_manager.class_counts == (1, 3, 1)


def test_simple_mode_switch_restores_legacy_buttons_and_saved_root(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "many_classes"
    for index in range(12):
        (root / f"class_{index:02d}").mkdir(parents=True)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    _open_simple_mode(window, root)
    qtbot.waitUntil(lambda: not window.active_tasks, timeout=5000)
    assert len(window.class_buttons) == 12
    assert all(button.isVisible() for button in window.class_buttons)

    window.mode_combo.setCurrentIndex(window.mode_combo.findData("label"))
    assert window.active_class_names == CLASS_NAMES
    assert all(button.isVisible() for button in window.class_buttons[:5])
    assert all(button.isHidden() for button in window.class_buttons[5:])
    assert window.canvas._annotation_enabled
    assert window.editor_class_filter_combo.isHidden()

    window.mode_combo.setCurrentIndex(
        window.mode_combo.findData("simple_class")
    )
    assert window.source_edit.text() == str(root.resolve())
    assert window.active_class_names == ()
    assert all(button.isHidden() for button in window.class_buttons)
    assert not window.canvas._annotation_enabled


def test_corrupt_image_can_be_skipped_deleted_and_restored(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    corrupt = root / "a" / "00_corrupt.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not-a-real-image")
    good = _image(root / "a" / "01_good.png", "green")
    errors = []

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    monkeypatch.setattr(window, "_show_error", errors.append)
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: bool(errors) and not window.active_tasks,
        timeout=5000,
    )

    assert window.current_path == corrupt
    assert window.navigate(1)
    qtbot.waitUntil(
        lambda: (
            window.current_path == good
            and not window.canvas._image.isNull()
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.navigate(-1)
    qtbot.waitUntil(
        lambda: len(errors) >= 2 and not window.active_tasks,
        timeout=5000,
    )
    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            not corrupt.exists()
            and corrupt not in window.classification_samples_by_path
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.undo_latest()
    qtbot.waitUntil(
        lambda: (
            corrupt.exists()
            and corrupt.read_bytes() == b"not-a-real-image"
            and corrupt in window.classification_samples_by_path
            and len(errors) >= 3
            and not window.active_tasks
        ),
        timeout=5000,
    )
    qtbot.wait(50)


def test_close_during_slow_image_failure_ignores_late_callback(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "data"
    _image(root / "a" / "one.png", "green")
    errors = []

    def slow_failure(path):
        time.sleep(0.25)
        raise RuntimeError("late decoder failure")

    monkeypatch.setattr(main_window_module, "load_qimage", slow_failure)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    monkeypatch.setattr(window, "_show_error", errors.append)
    _open_simple_mode(window, root)
    qtbot.waitUntil(
        lambda: (
            window.classification_index is not None
            and not window.busy
            and bool(window.active_tasks)
        ),
        timeout=5000,
    )

    assert window.close()
    assert window.closing
    qtbot.waitUntil(lambda: not window.active_tasks, timeout=5000)
    qtbot.wait(50)
    assert errors == []


def test_failed_rescan_preserves_open_dataset_and_controls(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    valid_root = tmp_path / "valid"
    valid_image = _image(valid_root / "a" / "one.png", "green")
    invalid_root = tmp_path / "invalid"
    _image(invalid_root / "orphan.png", "red")
    errors = []

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    monkeypatch.setattr(window, "_show_error", errors.append)
    _open_simple_mode(window, valid_root)
    qtbot.waitUntil(
        lambda: (
            window.current_path == valid_image
            and not window.active_tasks
            and not window.canvas.is_loading
        ),
        timeout=5000,
    )
    original_manager = window.classification_manager
    original_images = list(window.images)

    window.source_edit.setText(str(invalid_root))
    window.scan_source()
    qtbot.waitUntil(
        lambda: bool(errors) and not window.active_tasks,
        timeout=5000,
    )

    assert window.classification_manager is original_manager
    assert window.images == original_images
    assert window.current_path == valid_image
    assert valid_image.is_file()
    assert window.scan_button.isEnabled()
    assert window.mode_combo.isEnabled()
