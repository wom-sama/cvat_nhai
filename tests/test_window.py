from cvat_nhai.main_window import MainWindow


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
