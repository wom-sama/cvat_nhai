import time
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog

import cvat_nhai.main_window as main_window_module
from cvat_nhai.export_settings_dialog import ExportSettingsDialog
from cvat_nhai.main_window import MainWindow
from cvat_nhai.models import (
    BBox,
    ExportReport,
    YoloAnnotation,
    YoloDatasetIndex,
    YoloSample,
)
from cvat_nhai.schema import audit_schema
from cvat_nhai.utils import atomic_write_yaml


def test_main_window_smoke(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    assert window.windowTitle() == "CVAT Nhai"
    assert len(window.class_buttons) == 5
    window.select_class(4)
    assert window.selected_class == 4
    assert window.class_buttons[4].isChecked()


def test_scan_keeps_workers_alive_and_displays_first_image(
    qtbot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "incoming"
    nested = source / "nested"
    nested.mkdir(parents=True)
    for index in range(50):
        Image.new("RGB", (64, 48), (index, 100, 150)).save(
            nested / "image_{:03d}.jpg".format(index)
        )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.source_edit.setText(str(source))
    window.scan_source()

    qtbot.waitUntil(
        lambda: (
            len(window.images) == 50
            and not window.canvas._image.isNull()
            and window.scan_button.isEnabled()
            and window.scan_button.text() == "Quet anh"
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.current_path == sorted(
        source.rglob("*.jpg"),
        key=lambda value: str(value).casefold(),
    )[0]


def test_holding_a_or_d_continuously_navigates_and_stops_on_release(
    qtbot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "navigation"
    source.mkdir()
    for index in range(8):
        Image.new("RGB", (80, 60), (index * 20, 90, 120)).save(
            source / "image_{:02d}.jpg".format(index)
        )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.source_edit.setText(str(source))
    window.scan_source()
    qtbot.waitUntil(
        lambda: (
            len(window.images) == 8
            and not window.active_tasks
            and window.current_index == 0
        ),
        timeout=5000,
    )

    qtbot.keyPress(window, Qt.Key_D)
    qtbot.waitUntil(
        lambda: window.current_index >= 3,
        timeout=1500,
    )
    qtbot.keyRelease(window, Qt.Key_D)
    stopped_index = window.current_index
    qtbot.wait(400)
    assert window.current_index == stopped_index
    assert not window.navigation_timer.isActive()

    qtbot.keyPress(window, Qt.Key_A)
    qtbot.waitUntil(
        lambda: window.current_index < stopped_index,
        timeout=1000,
    )
    qtbot.keyRelease(window, Qt.Key_A)
    assert not window.navigation_timer.isActive()


def test_enter_keeps_window_visible_and_shows_busy_progress(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "current.jpg"
    Image.new("RGB", (120, 80), "green").save(source)

    class SlowManager:
        @staticmethod
        def annotate(path, bbox, class_id, requested_split):
            time.sleep(0.25)
            return type(
                "Result",
                (),
                {"action": "annotate", "split": "train"},
            )()

    monkeypatch.setattr(
        main_window_module,
        "audit_schema",
        lambda *args: type("Audit", (), {"ready": True})(),
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window.manager = SlowManager()
    window.images = [source]
    window.current_path = source
    window.current_index = 0
    window.selected_class = 0
    window.canvas.set_image(QImage(120, 80, QImage.Format_RGB32))
    window.canvas.set_bbox(BBox(10, 10, 100, 70))

    heartbeats = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: heartbeats.append(1))
    heartbeat.start()
    qtbot.keyClick(window, Qt.Key_Return)

    qtbot.waitUntil(lambda: window.busy_overlay.isVisible(), timeout=1000)
    assert window.isVisible()
    assert window.busy
    assert window.busy_overlay.progress_bar.maximum() == 0
    assert "Dang ghi hai dataset" in window.busy_overlay.detail_label.text()
    qtbot.waitUntil(lambda: not window.busy, timeout=3000)
    heartbeat.stop()
    assert len(heartbeats) >= 3
    assert window.isVisible()
    assert not window.busy_overlay.isVisible()


def test_prefetched_image_is_displayed_if_it_becomes_current(
    qtbot,
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefetched.jpg"
    Image.new("RGB", (320, 180), "orange").save(path)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.current_path = path
    window.canvas.set_image(QImage(40, 30, QImage.Format_RGB32))
    window.canvas.set_loading("Dang tai prefetched.jpg...")

    prefetched = QImage(320, 180, QImage.Format_RGB32)
    window._image_loaded(path, prefetched)

    assert window.canvas.image_size == (320, 180)
    assert not window.canvas.is_loading


def test_destination_root_creates_both_five_class_datasets(
    qtbot,
    tmp_path: Path,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    destination = tmp_path / "output"

    window.set_destination_root(destination)

    assert window.detection_root == destination / "dataset"
    assert window.classification_root == destination / "cls_crops"
    assert window.archive_root == destination / ".cvat_nhai_archive"
    assert (destination / "dataset" / "data.yaml").exists()
    assert (destination / "cls_crops" / "data.yaml").exists()
    assert (destination / "cls_crops" / "manifest.csv").exists()
    assert (destination / "cls_crops" / "stats.json").exists()
    assert audit_schema(
        window.detection_root,
        window.classification_root,
    ).ready


def test_export_dialog_reports_progress_without_blocking_ui(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "export"
    destination.mkdir()
    used_padding = []

    def slow_export(
        index,
        output,
        crop_padding,
        progress_callback=None,
        cancel_event=None,
    ):
        used_padding.append(crop_padding)
        for value in range(1, 8):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Export da bi huy an toan")
            progress_callback(
                {
                    "stage": "4/4 Dang ghi yolo_f va class_f",
                    "detail": "{} / 7".format(value),
                    "value": value,
                    "maximum": 7,
                }
            )
            time.sleep(0.04)
        return ExportReport(
            destination=Path(output),
            images=7,
            objects=7,
            skipped=0,
            class_counts={0: 7},
        )

    monkeypatch.setattr(
        main_window_module,
        "export_rebalanced_datasets",
        slow_export,
    )

    class FakeExportSettingsDialog:
        def __init__(self, preview_items, initial_padding, parent=None):
            self.crop_padding = 0.23

        def exec(self):
            return QDialog.DialogCode.Accepted

        def deleteLater(self):
            return None

    monkeypatch.setattr(
        main_window_module,
        "ExportSettingsDialog",
        FakeExportSettingsDialog,
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args: None,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    assert window.export_button.text() == "Xuat yolo_f + class_f"
    window.editor_index = YoloDatasetIndex(
        root=tmp_path / "source",
        data_yaml=tmp_path / "source" / "data.yaml",
        class_names=("class0",),
        samples=(),
    )
    monkeypatch.setattr(
        window,
        "_show_export_destination_dialog",
        lambda: window._start_editor_export(destination),
    )

    heartbeats = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: heartbeats.append(1))
    heartbeat.start()
    qtbot.mouseClick(window.export_button, Qt.LeftButton)

    qtbot.waitUntil(
        lambda: (
            window.export_progress_dialog is not None
            and window.export_progress_dialog.progress_bar.value() >= 3
        ),
        timeout=3000,
    )
    assert window.busy
    assert window.export_progress_dialog.isVisible()
    assert len(heartbeats) >= 3
    qtbot.waitUntil(lambda: not window.busy, timeout=3000)
    heartbeat.stop()
    assert window.export_progress_dialog is None
    assert used_padding == [0.23]
    assert window.export_crop_padding == 0.23
    assert float(window.settings.value("datasets/export_crop_padding")) == 0.23


def test_export_continue_opens_visible_non_native_folder_picker(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.editor_index = YoloDatasetIndex(
        root=tmp_path / "source",
        data_yaml=tmp_path / "source" / "data.yaml",
        class_names=("class0",),
        samples=(),
    )
    clicked = []

    def click_continue() -> None:
        modal = QApplication.activeModalWidget()
        if isinstance(modal, ExportSettingsDialog):
            clicked.append(True)
            qtbot.mouseClick(modal.continue_button, Qt.LeftButton)

    QTimer.singleShot(0, click_continue)
    window.export_editor_dataset()

    qtbot.waitUntil(
        lambda: (
            window.export_destination_dialog is not None
            and window.export_destination_dialog.isVisible()
        ),
        timeout=2000,
    )
    dialog = window.export_destination_dialog
    assert clicked == [True]
    assert isinstance(dialog, QFileDialog)
    assert dialog.testOption(QFileDialog.DontUseNativeDialog)
    assert dialog.fileMode() == QFileDialog.Directory
    assert dialog.windowModality() == Qt.WindowModal
    destination = tmp_path / "empty-export"
    destination.mkdir()
    started = []
    monkeypatch.setattr(
        window,
        "_start_editor_export",
        lambda path: started.append(path),
    )
    dialog.setDirectory(str(destination))
    dialog.accept()
    qtbot.waitUntil(lambda: bool(started), timeout=1000)
    assert started == [destination]
    qtbot.waitUntil(
        lambda: window.export_destination_dialog is None,
        timeout=1000,
    )


def test_editor_export_uses_full_dataset_when_split_filter_is_active(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "source"
    train_image = root / "images" / "train" / "a.jpg"
    val_image = root / "images" / "val" / "b.jpg"
    train_label = root / "labels" / "train" / "a.txt"
    val_label = root / "labels" / "val" / "b.txt"
    destination = tmp_path / "export"
    destination.mkdir()
    index = YoloDatasetIndex(
        root=root,
        data_yaml=root / "data.yaml",
        class_names=("class0",),
        samples=(
            YoloSample(train_image, train_label, "train"),
            YoloSample(val_image, val_label, "val"),
        ),
    )
    exported_paths = []

    def fake_export(
        exported_index,
        output,
        crop_padding,
        progress_callback=None,
        cancel_event=None,
    ):
        exported_paths.append(
            tuple(sample.image_path for sample in exported_index.samples)
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "4/4 Dang ghi yolo_f va class_f",
                    "detail": "done",
                    "value": 1,
                    "maximum": 1,
                }
            )
        return ExportReport(
            destination=Path(output),
            images=2,
            objects=2,
            skipped=0,
            class_counts={0: 2},
        )

    monkeypatch.setattr(
        main_window_module,
        "export_rebalanced_datasets",
        fake_export,
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda *args: None,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.editor_index = index
    window.editor_manager = object()
    window.editor_samples_by_path = {
        sample.image_path: sample for sample in index.samples
    }
    window.editor_all_images = [train_image, val_image]
    window._set_editor_split_filter_ui("val")
    window.images = [val_image]

    window._start_editor_export(destination)
    qtbot.waitUntil(lambda: not window.busy, timeout=3000)

    assert exported_paths == [(train_image, val_image)]


def test_yolo_editor_mode_loads_resets_saves_and_deletes(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "old_yolo"
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
            "nc": 3,
            "names": ["green", "ripe", "bad"],
        },
    )
    first_image = root / "images" / "train" / "a.jpg"
    second_image = root / "images" / "val" / "b.jpg"
    Image.new("RGB", (200, 100), "green").save(first_image)
    Image.new("RGB", (100, 100), "orange").save(second_image)
    first_label = root / "labels" / "train" / "a.txt"
    second_label = root / "labels" / "val" / "b.txt"
    first_label.write_text(
        "0 0.250000 0.500000 0.400000 0.600000\n"
        "1 0.750000 0.500000 0.400000 0.600000\n",
        encoding="utf-8",
    )
    second_label.write_text(
        "2 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.source_edit.setText(str(root))
    window.scan_source()
    qtbot.waitUntil(
        lambda: (
            len(window.images) == 2
            and len(window.canvas.annotations) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )

    starting_box = window.canvas.active_index
    window.class_buttons[0].setFocus()
    qtbot.keyClick(window.class_buttons[0], Qt.Key_Tab)
    assert window.canvas.active_index == (starting_box + 1) % 2
    qtbot.keyClick(
        window.class_buttons[0],
        Qt.Key_Tab,
        modifier=Qt.ShiftModifier,
    )
    assert window.canvas.active_index == starting_box

    changed = (
        YoloAnnotation(2, BBox(20, 10, 180, 90)),
        window.canvas.annotations[1],
    )
    window.canvas.set_annotations(changed, 0)
    window.commit_current()
    qtbot.waitUntil(
        lambda: (
            window.current_path == second_image
            and not window.active_tasks
            and len(window.canvas.annotations) == 1
        ),
        timeout=5000,
    )
    assert first_label.read_text(encoding="utf-8").splitlines()[0].startswith(
        "2 0.500000 0.500000 0.800000 0.800000"
    )

    window.select_class(0)
    assert window.canvas.annotations[0].class_id == 0
    current_before_navigation = window.current_path
    window.navigate(-1)
    assert window.current_path == current_before_navigation
    window.seek_to_index(0)
    assert window.current_path == current_before_navigation
    assert window.progress.value() == window.current_index
    window.reset_annotation()
    assert window.canvas.annotations[0].class_id == 2

    window.seek_to_index(0)
    qtbot.waitUntil(
        lambda: (
            window.current_path == first_image
            and len(window.canvas.annotations) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.seek_to_index(1)
    qtbot.waitUntil(
        lambda: (
            window.current_path == second_image
            and len(window.canvas.annotations) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )

    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            len(window.images) == 1
            and not second_image.exists()
            and not second_label.exists()
            and not window.active_tasks
        ),
        timeout=5000,
    )


def test_yolo_editor_shows_and_filters_dataset_splits(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "split_yolo"
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
            "nc": 2,
            "names": ["green", "ripe"],
        },
    )
    train_image = root / "images" / "train" / "train_a.jpg"
    val_image = root / "images" / "val" / "val_b.jpg"
    Image.new("RGB", (180, 120), "green").save(train_image)
    Image.new("RGB", (160, 120), "orange").save(val_image)
    (root / "labels" / "train" / "train_a.txt").write_text(
        "0 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )
    val_label = root / "labels" / "val" / "val_b.txt"
    val_label.write_text(
        "1 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.source_edit.setText(str(root))
    window.scan_source()
    qtbot.waitUntil(
        lambda: (
            window.current_path == train_image
            and len(window.canvas.annotations) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.editor_split_filter_combo.isVisible()
    assert window.editor_split_status.text().startswith(
        "Split hien tai: TRAIN"
    )
    assert window.queue_label.text() == "2 anh dataset"

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("val")
    )
    qtbot.waitUntil(
        lambda: (
            window.current_path == val_image
            and len(window.images) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.editor_split_status.text().startswith(
        "Split hien tai: VAL"
    )
    assert window.queue_label.text() == "1 anh VAL / 2 tong"
    assert window.progress.maximum() == 0

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("test")
    )
    qtbot.waitUntil(lambda: window.current_path is None, timeout=1000)
    assert window.images == []
    assert window.file_label.text() == "Khong co anh trong bo loc TEST"
    assert window.queue_label.text() == "0 anh TEST / 2 tong"
    assert window.editor_split_status.text() == "Split hien tai: -"

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("all")
    )
    qtbot.waitUntil(
        lambda: (
            window.current_path == train_image
            and len(window.canvas.annotations) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.canvas.set_annotations(
        (YoloAnnotation(1, BBox(20, 20, 120, 90)),),
        0,
    )
    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("val")
    )
    assert window.editor_split_filter == "all"
    assert window.editor_split_filter_combo.currentData() == "all"
    assert window.current_path == train_image

    window.reset_annotation()
    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("val")
    )
    qtbot.waitUntil(
        lambda: (
            window.current_path == val_image
            and len(window.canvas.annotations) == 1
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.delete_current()
    qtbot.waitUntil(
        lambda: (
            len(window.editor_all_images) == 1
            and not val_image.exists()
            and not val_label.exists()
            and window.current_path is None
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.queue_label.text() == "0 anh VAL / 1 tong"
    assert val_image not in window.editor_samples_by_path

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("all")
    )
    qtbot.waitUntil(
        lambda: window.current_path == train_image,
        timeout=5000,
    )
    assert window.images == [train_image]


def test_yolo_editor_filters_by_class_and_updates_after_save(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "class_filter_yolo"
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
            "nc": 2,
            "names": ["green", "ripe"],
        },
    )
    train_image = root / "images" / "train" / "mix.jpg"
    val_image = root / "images" / "val" / "ripe.jpg"
    test_image = root / "images" / "test" / "green.jpg"
    Image.new("RGB", (200, 120), "green").save(train_image)
    Image.new("RGB", (180, 120), "orange").save(val_image)
    Image.new("RGB", (160, 120), "yellow").save(test_image)
    train_label = root / "labels" / "train" / "mix.txt"
    train_label.write_text(
        "0 0.250000 0.500000 0.300000 0.500000\n"
        "1 0.750000 0.500000 0.300000 0.500000\n",
        encoding="utf-8",
    )
    (root / "labels" / "val" / "ripe.txt").write_text(
        "1 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )
    (root / "labels" / "test" / "green.txt").write_text(
        "0 0.500000 0.500000 0.500000 0.500000\n",
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.source_edit.setText(str(root))
    window.scan_source()
    qtbot.waitUntil(
        lambda: (
            window.editor_index is not None
            and window.current_path is not None
            and not window.active_tasks
        ),
        timeout=5000,
    )

    assert window.editor_class_filter_combo.findData(1) >= 0
    window.editor_class_filter_combo.setCurrentIndex(
        window.editor_class_filter_combo.findData(1)
    )
    qtbot.waitUntil(
        lambda: (
            len(window.images) == 2
            and train_image in window.images
            and val_image in window.images
            and test_image not in window.images
            and window.current_path == train_image
            and len(window.canvas.annotations) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert "class 2" in window.queue_label.text()
    assert window.editor_split_status.text().endswith("class 2 (ripe)")

    window.canvas.set_annotations(
        (YoloAnnotation(0, BBox(20, 20, 120, 90)),),
        0,
    )
    window.editor_class_filter_combo.setCurrentIndex(
        window.editor_class_filter_combo.findData("all")
    )
    assert window.editor_class_filter == 1
    assert window.editor_class_filter_combo.currentData() == 1
    window.reset_annotation()

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("train")
    )
    qtbot.waitUntil(
        lambda: (
            window.images == [train_image]
            and window.current_path == train_image
            and len(window.canvas.annotations) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )
    window.canvas.set_annotations(
        (
            YoloAnnotation(0, BBox(20, 30, 80, 90)),
            YoloAnnotation(0, BBox(120, 30, 180, 90)),
        ),
        0,
    )
    window.commit_current()
    qtbot.waitUntil(
        lambda: (
            window.current_path is None
            and not window.images
            and not window.active_tasks
        ),
        timeout=5000,
    )

    assert window.editor_image_classes[train_image] == frozenset({0})
    assert all(
        line.startswith("0 ")
        for line in train_label.read_text(
            encoding="utf-8"
        ).splitlines()
    )
    assert "class 2" in window.queue_label.text()

    window.editor_split_filter_combo.setCurrentIndex(
        window.editor_split_filter_combo.findData("all")
    )
    qtbot.waitUntil(
        lambda: (
            window.images == [val_image]
            and window.current_path == val_image
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert train_image not in window.images


def test_yolo_editor_loads_tiny_box_and_selects_largest(
    qtbot,
    tmp_path: Path,
) -> None:
    root = tmp_path / "tiny_yolo"
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
            "nc": 5,
            "names": [
                "c0",
                "c1",
                "c2",
                "c3",
                "c4",
            ],
        },
    )
    image = root / "images" / "train" / "tiny.jpg"
    Image.new("RGB", (640, 640), "green").save(image)
    (root / "labels" / "train" / "tiny.txt").write_text(
        "4 0.032531 0.144258 0.002156 0.002172\n"
        "4 0.514641 0.505453 0.970719 0.715906\n",
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.mode_combo.setCurrentIndex(1)
    window.source_edit.setText(str(root))
    window.scan_source()

    qtbot.waitUntil(
        lambda: (
            window.current_path == image
            and len(window.canvas.annotations) == 2
            and not window.active_tasks
        ),
        timeout=5000,
    )
    assert window.canvas.active_index == 1
    assert window.canvas.annotations[0].bbox.width < 3
    assert not window._editor_is_dirty()
