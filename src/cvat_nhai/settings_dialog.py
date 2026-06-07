from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PathRow(QWidget):
    def __init__(self, value: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(value)
        button = QPushButton("...")
        button.setFixedWidth(40)
        button.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(button)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chon thu muc",
            self.edit.text(),
        )
        if folder:
            self.edit.setText(folder)


class SettingsDialog(QDialog):
    def __init__(
        self,
        detection_root: Path,
        classification_root: Path,
        archive_root: Path,
        crop_padding: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cau hinh dataset")
        self.setMinimumWidth(680)

        self.detection_row = PathRow(str(detection_root))
        self.classification_row = PathRow(str(classification_root))
        self.archive_row = PathRow(str(archive_root))
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 0.5)
        self.padding_spin.setSingleStep(0.01)
        self.padding_spin.setDecimals(2)
        self.padding_spin.setValue(crop_padding)
        self.padding_spin.setToolTip(
            "Ty le mo rong moi canh cua bbox truoc khi letterbox 640x640"
        )

        form = QFormLayout()
        form.addRow("YOLO detection:", self.detection_row)
        form.addRow("Classification crops:", self.classification_row)
        form.addRow("Safety archive:", self.archive_row)
        form.addRow("Crop padding moi canh:", self.padding_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, object]:
        return {
            "detection_root": Path(self.detection_row.edit.text().strip()),
            "classification_root": Path(
                self.classification_row.edit.text().strip()
            ),
            "archive_root": Path(self.archive_row.edit.text().strip()),
            "crop_padding": self.padding_spin.value(),
        }
