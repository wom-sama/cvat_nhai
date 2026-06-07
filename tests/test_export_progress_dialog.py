from PIL import Image, ImageDraw

from cvat_nhai.export_progress_dialog import ExportProgressDialog
from cvat_nhai.export_settings_dialog import (
    ExportPreviewItem,
    ExportSettingsDialog,
)
from cvat_nhai.models import BBox


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


def test_export_settings_dialog_updates_margin_preview_realtime(
    qtbot,
) -> None:
    image = Image.new("RGB", (200, 120), "#27364A")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 30, 140, 90), fill="#F59E0B")
    item = ExportPreviewItem(
        image=image,
        bbox=BBox(60, 30, 140, 90),
        image_name="sample.jpg",
        class_name="ripe",
    )
    dialog = ExportSettingsDialog([item], 0.08)
    qtbot.addWidget(dialog)
    dialog.show()

    original_key = dialog.cards[0].image_label.pixmap().cacheKey()
    dialog.padding_slider.setValue(25)

    assert dialog.crop_padding == 0.25
    assert dialog.padding_spin.value() == 0.25
    assert dialog.percent_label.text() == "25% moi canh"
    assert not dialog.cards[0].image_label.pixmap().isNull()
    assert dialog.cards[0].image_label.pixmap().cacheKey() != original_key
