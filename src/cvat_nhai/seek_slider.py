from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QSlider, QWidget


class SeekSlider(QSlider):
    seek_requested = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(Qt.Horizontal, parent)
        self.setTracking(True)
        self.setSingleStep(1)
        self.setPageStep(100)
        self.setCursor(Qt.PointingHandCursor)
        self._dragging = False

    def _value_from_x(self, x: float) -> int:
        if self.width() <= 1 or self.maximum() <= self.minimum():
            return self.minimum()
        ratio = max(0.0, min(1.0, x / float(self.width() - 1)))
        return round(
            self.minimum()
            + ratio * (self.maximum() - self.minimum())
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            value = self._value_from_x(event.position().x())
            self.setValue(value)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.LeftButton:
            value = self._value_from_x(event.position().x())
            self.setValue(value)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            value = self._value_from_x(event.position().x())
            self.setValue(value)
            self.seek_requested.emit(value)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        previous = self.value()
        super().keyPressEvent(event)
        if self.value() != previous:
            self.seek_requested.emit(self.value())
