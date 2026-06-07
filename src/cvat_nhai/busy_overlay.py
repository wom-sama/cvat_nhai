from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class BusyOverlay(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet(
            """
            BusyOverlay {
                background: rgba(2, 6, 23, 95);
            }
            QFrame#busyCard {
                background: #111827;
                border: 1px solid #475569;
                border-radius: 10px;
            }
            QLabel#busyTitle {
                color: #F8FAFC;
                font-size: 13pt;
                font-weight: 600;
            }
            QLabel#busyDetail {
                color: #CBD5E1;
            }
            QProgressBar {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 5px;
                min-height: 12px;
                max-height: 12px;
            }
            QProgressBar::chunk {
                background: #0EA5E9;
                border-radius: 4px;
            }
            """
        )

        card = QFrame()
        card.setObjectName("busyCard")
        card.setFixedWidth(440)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.setSpacing(10)

        self.title_label = QLabel("Dang xu ly")
        self.title_label.setObjectName("busyTitle")
        self.detail_label = QLabel()
        self.detail_label.setObjectName("busyDetail")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)

        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.detail_label)
        card_layout.addWidget(self.progress_bar)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addStretch()
        layout.addWidget(card, 0, Qt.AlignHCenter)
        layout.addStretch()
        self.hide()

    def show_message(self, message: str) -> None:
        self.detail_label.setText(message or "Vui long doi trong giay lat...")
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
