from pathlib import Path

from PIL import Image

from cvat_nhai.main_window import MainWindow
from cvat_nhai.schema import audit_schema


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
