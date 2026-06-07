from cvat_nhai.export_progress_dialog import ExportProgressDialog


def test_export_progress_dialog_updates_and_requests_cancel(qtbot) -> None:
    dialog = ExportProgressDialog()
    qtbot.addWidget(dialog)
    requested = []
    dialog.cancel_requested.connect(lambda: requested.append(True))
    dialog.show()

    dialog.update_progress(
        {
            "stage": "1/4 Dang doc anh",
            "detail": "25 / 100",
            "value": 25,
            "maximum": 100,
        }
    )

    assert dialog.progress_bar.value() == 25
    assert dialog.progress_bar.maximum() == 100
    dialog.request_cancel()
    assert requested == [True]
    assert not dialog.cancel_button.isEnabled()
    dialog.finish()
