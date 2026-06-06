from typing import Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .models import BBox


class AnnotationCanvas(QWidget):
    bbox_changed = Signal(object)

    HANDLE_SIZE = 9.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 320)

        self._image = QImage()
        self._pixmap = QPixmap()
        self._bbox: Optional[BBox] = None
        self._class_name = ""
        self._class_color = QColor("#22C55E")
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._space_down = False
        self._mode = ""
        self._handle = ""
        self._start_widget = QPointF()
        self._start_image = QPointF()
        self._original_bbox: Optional[BBox] = None
        self.setCursor(Qt.CrossCursor)

    @property
    def bbox(self) -> Optional[BBox]:
        return self._bbox

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image.width(), self._image.height()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._pixmap = QPixmap.fromImage(image)
        self._bbox = None
        self.fit_to_view()
        self.update()

    def clear_image(self) -> None:
        self._image = QImage()
        self._pixmap = QPixmap()
        self._bbox = None
        self.update()

    def set_bbox(self, bbox: Optional[BBox]) -> None:
        if bbox is not None and not self._image.isNull():
            bbox = bbox.clamp(self._image.width(), self._image.height())
        self._bbox = bbox
        self.bbox_changed.emit(bbox)
        self.update()

    def reset_annotation(self) -> None:
        self.set_bbox(None)

    def set_class_style(self, name: str, color: str) -> None:
        self._class_name = name
        self._class_color = QColor(color)
        self.update()

    def fit_to_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _base_scale(self) -> float:
        if self._image.isNull():
            return 1.0
        margin = 24.0
        available_w = max(1.0, self.width() - margin * 2)
        available_h = max(1.0, self.height() - margin * 2)
        return min(
            available_w / self._image.width(),
            available_h / self._image.height(),
        )

    def _scale(self) -> float:
        return max(0.001, self._base_scale() * self._zoom)

    def _image_rect(self) -> QRectF:
        if self._image.isNull():
            return QRectF()
        scale = self._scale()
        width = self._image.width() * scale
        height = self._image.height() * scale
        left = (self.width() - width) / 2.0 + self._pan.x()
        top = (self.height() - height) / 2.0 + self._pan.y()
        return QRectF(left, top, width, height)

    def _widget_to_image(self, point: QPointF, clamp: bool = True) -> QPointF:
        rect = self._image_rect()
        scale = self._scale()
        x = (point.x() - rect.left()) / scale
        y = (point.y() - rect.top()) / scale
        if clamp and not self._image.isNull():
            x = max(0.0, min(float(self._image.width()), x))
            y = max(0.0, min(float(self._image.height()), y))
        return QPointF(x, y)

    def _image_to_widget(self, point: QPointF) -> QPointF:
        rect = self._image_rect()
        scale = self._scale()
        return QPointF(
            rect.left() + point.x() * scale,
            rect.top() + point.y() * scale,
        )

    def _bbox_widget_rect(self) -> QRectF:
        if self._bbox is None:
            return QRectF()
        top_left = self._image_to_widget(
            QPointF(self._bbox.x1, self._bbox.y1)
        )
        bottom_right = self._image_to_widget(
            QPointF(self._bbox.x2, self._bbox.y2)
        )
        return QRectF(top_left, bottom_right).normalized()

    def _handle_points(self) -> dict:
        rect = self._bbox_widget_rect()
        if rect.isNull():
            return {}
        center = rect.center()
        return {
            "nw": rect.topLeft(),
            "n": QPointF(center.x(), rect.top()),
            "ne": rect.topRight(),
            "e": QPointF(rect.right(), center.y()),
            "se": rect.bottomRight(),
            "s": QPointF(center.x(), rect.bottom()),
            "sw": rect.bottomLeft(),
            "w": QPointF(rect.left(), center.y()),
        }

    def _hit_handle(self, point: QPointF) -> str:
        radius = self.HANDLE_SIZE + 4.0
        for name, handle_point in self._handle_points().items():
            if (
                abs(point.x() - handle_point.x()) <= radius
                and abs(point.y() - handle_point.y()) <= radius
            ):
                return name
        return ""

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0B1120"))

        if self._image.isNull():
            painter.setPen(QColor("#94A3B8"))
            font = QFont()
            font.setPointSize(13)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Chon thu muc anh de bat dau",
            )
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        target = self._image_rect()
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        if self._bbox is None:
            return

        box_rect = self._bbox_widget_rect()
        pen = QPen(self._class_color, 2.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(self._class_color)
        fill.setAlpha(28)
        painter.setBrush(fill)
        painter.drawRect(box_rect)

        label = self._class_name or "BBox"
        metrics = painter.fontMetrics()
        label_width = metrics.horizontalAdvance(label) + 16
        label_height = metrics.height() + 8
        label_top = max(target.top(), box_rect.top() - label_height)
        label_rect = QRectF(
            box_rect.left(),
            label_top,
            label_width,
            label_height,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._class_color)
        painter.drawRoundedRect(label_rect, 4, 4)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(
            label_rect.adjusted(8, 0, -8, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            label,
        )

        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.setBrush(self._class_color)
        size = self.HANDLE_SIZE
        for point in self._handle_points().values():
            painter.drawRect(
                QRectF(
                    point.x() - size / 2.0,
                    point.y() - size / 2.0,
                    size,
                    size,
                )
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image.isNull():
            return
        point = event.position()
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self._space_down
        ):
            self._mode = "pan"
            self._start_widget = point
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return

        self.setFocus()
        handle = self._hit_handle(point)
        self._start_widget = point
        self._start_image = self._widget_to_image(point)
        self._original_bbox = self._bbox
        if handle and self._bbox is not None:
            self._mode = "resize"
            self._handle = handle
        elif (
            self._bbox is not None
            and self._bbox_widget_rect().contains(point)
        ):
            self._mode = "move"
        elif self._image_rect().contains(point):
            self._mode = "draw"
            self._bbox = BBox(
                self._start_image.x(),
                self._start_image.y(),
                self._start_image.x(),
                self._start_image.y(),
            )
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = event.position()
        if self._mode == "pan":
            delta = point - self._start_widget
            self._pan += delta
            self._start_widget = point
            self.update()
            return
        if self._mode == "":
            handle = self._hit_handle(point)
            if handle:
                if handle in {"nw", "se"}:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif handle in {"ne", "sw"}:
                    self.setCursor(Qt.SizeBDiagCursor)
                elif handle in {"n", "s"}:
                    self.setCursor(Qt.SizeVerCursor)
                else:
                    self.setCursor(Qt.SizeHorCursor)
            elif (
                self._bbox is not None
                and self._bbox_widget_rect().contains(point)
            ):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            return

        current = self._widget_to_image(point)
        width = self._image.width()
        height = self._image.height()
        if self._mode == "draw":
            self._bbox = BBox(
                self._start_image.x(),
                self._start_image.y(),
                current.x(),
                current.y(),
            ).normalized().clamp(width, height)
        elif self._mode == "move" and self._original_bbox is not None:
            dx = current.x() - self._start_image.x()
            dy = current.y() - self._start_image.y()
            box = self._original_bbox
            dx = max(-box.x1, min(width - box.x2, dx))
            dy = max(-box.y1, min(height - box.y2, dy))
            self._bbox = BBox(
                box.x1 + dx,
                box.y1 + dy,
                box.x2 + dx,
                box.y2 + dy,
            )
        elif self._mode == "resize" and self._original_bbox is not None:
            box = self._original_bbox
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            if "w" in self._handle:
                x1 = current.x()
            if "e" in self._handle:
                x2 = current.x()
            if "n" in self._handle:
                y1 = current.y()
            if "s" in self._handle:
                y2 = current.y()
            self._bbox = BBox(x1, y1, x2, y2).normalized().clamp(
                width,
                height,
            )
        self.bbox_changed.emit(self._bbox)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode in {"draw", "move", "resize"}:
            if self._bbox is not None and not self._bbox.is_valid():
                self._bbox = None
            self.bbox_changed.emit(self._bbox)
        self._mode = ""
        self._handle = ""
        self._original_bbox = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image.isNull():
            return
        cursor = event.position()
        image_point = self._widget_to_image(cursor, clamp=False)
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = max(0.2, min(12.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 0.0001:
            return
        self._zoom = new_zoom
        mapped = self._image_to_widget(image_point)
        self._pan += cursor - mapped
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.fit_to_view()
