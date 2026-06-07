from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt

from cvat_nhai.main_window import MainWindow
from cvat_nhai.models import BBox, YoloAnnotation
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
