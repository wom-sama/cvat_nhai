from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class ExportProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = True
        self._cancel_requested = False
        self.setWindowTitle("Dang xuat dataset")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(540)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setStyleSheet(
            """
            QDialog {
                background: #111827;
                color: #E5E7EB;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            QLabel { color: #E5E7EB; }
            QProgressBar {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #F8FAFC;
                min-height: 22px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #0EA5E9;
                border-radius: 5px;
            }
            QPushButton {
                background: #1E293B;
                border: 1px solid #475569;
                border-radius: 7px;
                color: #F8FAFC;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background: #26364D;
                border-color: #64748B;
            }
            QPushButton:disabled { color: #64748B; }
            """
        )

        self.stage_label = QLabel("Dang chuan bi...")
        self.stage_label.setStyleSheet("font-weight: 600;")
        self.detail_label = QLabel(
            "Ung dung van dang lam viec. Khong tat chuong trinh."
        )
        self.detail_label.setObjectName("muted")
        self.detail_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.cancel_button = QPushButton("Huy an toan")
        self.cancel_button.clicked.connect(self.request_cancel)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.detail_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(button_row)

    def update_progress(self, payload: object) -> None:
        if not isinstance(payload, dict) or self._cancel_requested:
            return
        self.stage_label.setText(
            str(payload.get("stage", "Dang xu ly..."))
        )
        self.detail_label.setText(str(payload.get("detail", "")))
        value = int(payload.get("value", 0))
        maximum = int(payload.get("maximum", 0))
        if maximum > 0:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(max(0, min(value, maximum)))
        else:
            self.progress_bar.setRange(0, 0)

    def request_cancel(self) -> None:
        if not self._running or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.stage_label.setText("Dang huy an toan...")
        self.detail_label.setText(
            "Dang don thu muc tam. Dataset nguon khong bi thay doi."
        )
        self.progress_bar.setRange(0, 0)
        self.cancel_requested.emit()

    def finish(self) -> None:
        self._running = False
        self.close()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._running:
            self.request_cancel()
            event.ignore()
            return
        super().closeEvent(event)
