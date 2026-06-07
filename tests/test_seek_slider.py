from PySide6.QtCore import QPoint, Qt

from cvat_nhai.seek_slider import SeekSlider


def test_seek_slider_emits_only_when_drag_is_released(qtbot) -> None:
    slider = SeekSlider()
    slider.resize(500, 20)
    slider.setRange(0, 999)
    slider.show()
    qtbot.addWidget(slider)
    qtbot.waitExposed(slider)
    requested = []
    slider.seek_requested.connect(requested.append)

    qtbot.mousePress(slider, Qt.LeftButton, pos=QPoint(50, 10))
    qtbot.mouseMove(slider, QPoint(400, 10))
    assert requested == []
    qtbot.mouseRelease(slider, Qt.LeftButton, pos=QPoint(400, 10))

    assert len(requested) == 1
    assert requested[0] == slider.value()
    assert 790 <= slider.value() <= 810


def test_seek_slider_keyboard_requests_new_position(qtbot) -> None:
    slider = SeekSlider()
    slider.setRange(0, 999)
    slider.setValue(500)
    slider.show()
    qtbot.addWidget(slider)
    slider.setFocus()
    requested = []
    slider.seek_requested.connect(requested.append)

    qtbot.keyClick(slider, Qt.Key_PageUp)

    assert requested == [600]
