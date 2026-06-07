import random
from dataclasses import dataclass
from typing import Sequence

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .image_ops import classification_crop
from .models import BBox


@dataclass(frozen=True)
class ExportPreviewItem:
    image: Image.Image
    bbox: BBox
    image_name: str
    class_name: str


class ExportPreviewCard(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(260, 260)
        self.image_label.setStyleSheet(
            "background:#0B1120;border:1px solid #334155;border-radius:7px;"
        )
        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color:#CBD5E1;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.image_label)
        layout.addWidget(self.caption)

    def set_item(self, item: ExportPreviewItem, padding: float) -> None:
        preview = classification_crop(
            item.image,
            item.bbox,
            padding,
            output_size=320,
        )
        data = preview.tobytes("raw", "RGB")
        qimage = QImage(
            data,
            preview.width,
            preview.height,
            preview.width * 3,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.caption.setText(
            "{}\n{}".format(item.image_name, item.class_name)
        )


class ExportSettingsDialog(QDialog):
    def __init__(
        self,
        preview_items: Sequence[ExportPreviewItem],
        initial_padding: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.preview_items = list(preview_items)
        self.visible_items = []
        self.cards = [ExportPreviewCard() for _ in range(4)]
        self.setWindowTitle("Cau hinh class_f")
        self.setMinimumSize(760, 760)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(
            """
            QDialog, QWidget {
                background:#111827;
                color:#E5E7EB;
                font-family:"Segoe UI";
                font-size:10pt;
            }
            QDoubleSpinBox {
                background:#182235;
                border:1px solid #334155;
                border-radius:7px;
                padding:7px;
                color:#F8FAFC;
            }
            QPushButton {
                background:#1E293B;
                border:1px solid #475569;
                border-radius:7px;
                padding:8px 12px;
                color:#F8FAFC;
            }
            QPushButton:hover { background:#26364D; }
            QSlider::groove:horizontal {
                background:#1E293B;height:7px;border-radius:3px;
            }
            QSlider::sub-page:horizontal {
                background:#0EA5E9;border-radius:3px;
            }
            QSlider::handle:horizontal {
                background:#E2E8F0;border:1px solid #0EA5E9;
                width:15px;margin:-5px 0;border-radius:7px;
            }
            """
        )

        title = QLabel("Margin quanh bbox khi tao class_f")
        title.setStyleSheet("font-size:14pt;font-weight:600;")
        hint = QLabel(
            "Margin duoc tinh theo phan tram moi canh bbox. Preview ben duoi "
            "la ket qua crop + letterbox; file xuat thuc te co kich thuoc 640x640."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94A3B8;")

        self.padding_slider = QSlider(Qt.Horizontal)
        self.padding_slider.setRange(0, 50)
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 0.5)
        self.padding_spin.setDecimals(2)
        self.padding_spin.setSingleStep(0.01)
        self.padding_spin.setSuffix("  (moi canh)")
        initial = max(0.0, min(0.5, float(initial_padding)))
        self.padding_slider.setValue(round(initial * 100))
        self.padding_spin.setValue(initial)
        self.percent_label = QLabel()
        self.percent_label.setMinimumWidth(110)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Margin:"))
        control_row.addWidget(self.padding_slider, 1)
        control_row.addWidget(self.percent_label)
        control_row.addWidget(self.padding_spin)

        self.sample_button = QPushButton("Doi mau ngau nhien")
        self.sample_button.clicked.connect(self.choose_random_items)
        self.preview_status = QLabel()
        self.preview_status.setStyleSheet("color:#94A3B8;")
        sample_row = QHBoxLayout()
        sample_row.addWidget(self.preview_status)
        sample_row.addStretch()
        sample_row.addWidget(self.sample_button)

        preview_grid = QGridLayout()
        preview_grid.setSpacing(12)
        for index, card in enumerate(self.cards):
            preview_grid.addWidget(card, index // 2, index % 2)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.continue_button = self.button_box.button(
            QDialogButtonBox.Save
        )
        self.continue_button.setText(
            "Tiep tuc chon thu muc xuat"
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(control_row)
        layout.addLayout(sample_row)
        layout.addLayout(preview_grid)
        layout.addWidget(self.button_box)

        self.padding_slider.valueChanged.connect(
            self._slider_changed
        )
        self.padding_spin.valueChanged.connect(self._spin_changed)
        self.choose_random_items()
        self._update_padding_text()

    @property
    def crop_padding(self) -> float:
        return float(self.padding_spin.value())

    def _slider_changed(self, value: int) -> None:
        desired = value / 100.0
        if abs(self.padding_spin.value() - desired) > 0.0001:
            self.padding_spin.blockSignals(True)
            self.padding_spin.setValue(desired)
            self.padding_spin.blockSignals(False)
        self._update_padding_text()
        self.update_previews()

    def _spin_changed(self, value: float) -> None:
        desired = round(value * 100)
        if self.padding_slider.value() != desired:
            self.padding_slider.blockSignals(True)
            self.padding_slider.setValue(desired)
            self.padding_slider.blockSignals(False)
        self._update_padding_text()
        self.update_previews()

    def _update_padding_text(self) -> None:
        self.percent_label.setText(
            "{:.0f}% moi canh".format(self.crop_padding * 100)
        )

    def choose_random_items(self) -> None:
        if not self.preview_items:
            self.visible_items = []
            self.preview_status.setText(
                "Khong tim thay bbox hop le de xem truoc"
            )
            self.sample_button.setEnabled(False)
            for card in self.cards:
                card.hide()
            return
        count = min(len(self.cards), len(self.preview_items))
        self.visible_items = random.sample(self.preview_items, count)
        self.preview_status.setText(
            "Dang xem truoc {} bbox ngau nhien".format(count)
        )
        for index, card in enumerate(self.cards):
            card.setVisible(index < count)
        self.update_previews()

    def update_previews(self) -> None:
        for card, item in zip(self.cards, self.visible_items):
            card.set_item(item, self.crop_padding)
