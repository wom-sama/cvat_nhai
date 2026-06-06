from typing import List, Optional, Sequence, Tuple

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

from .models import BBox, YoloAnnotation


class AnnotationCanvas(QWidget):
    bbox_changed = Signal(object)
    annotations_changed = Signal(object)
    active_annotation_changed = Signal(int)

    HANDLE_SIZE = 9.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 320)

        self._image = QImage()
        self._pixmap = QPixmap()
        self._bbox: Optional[BBox] = None
        self._annotations: List[YoloAnnotation] = []
        self._active_index = -1
        self._multi_mode = False
        self._class_names: Tuple[str, ...] = ()
        self._class_colors: Tuple[str, ...] = ()
        self._current_class_id = 0
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
    def annotations(self) -> Tuple[YoloAnnotation, ...]:
        return tuple(self._annotations)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image.width(), self._image.height()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._pixmap = QPixmap.fromImage(image)
        self._bbox = None
        self._annotations = []
        self._active_index = -1
        self._multi_mode = False
        self.fit_to_view()
        self.update()

    def clear_image(self) -> None:
        self._image = QImage()
        self._pixmap = QPixmap()
        self._bbox = None
        self._annotations = []
        self._active_index = -1
        self._multi_mode = False
        self.update()

    def set_bbox(self, bbox: Optional[BBox]) -> None:
        if bbox is not None and not self._image.isNull():
            bbox = bbox.clamp(self._image.width(), self._image.height())
        self._bbox = bbox
        self.bbox_changed.emit(bbox)
        self.update()

    def set_class_catalog(
        self,
        names: Sequence[str],
        colors: Sequence[str],
    ) -> None:
        self._class_names = tuple(str(name) for name in names)
        self._class_colors = tuple(str(color) for color in colors)
        self.update()

    def set_annotations(
        self,
        annotations: Sequence[YoloAnnotation],
        active_index: int = 0,
    ) -> None:
        self._multi_mode = True
        self._annotations = list(annotations)
        if self._annotations:
            self._active_index = max(
                0,
                min(active_index, len(self._annotations) - 1),
            )
            self._bbox = self._annotations[self._active_index].bbox
            self._current_class_id = self._annotations[
                self._active_index
            ].class_id
        else:
            self._active_index = -1
            self._bbox = None
        self.bbox_changed.emit(self._bbox)
        self.active_annotation_changed.emit(self._active_index)
        self.annotations_changed.emit(tuple(self._annotations))
        self.update()

    def set_active_annotation(self, index: int) -> None:
        if not self._annotations:
            self._active_index = -1
            self._bbox = None
        else:
            self._active_index = index % len(self._annotations)
            annotation = self._annotations[self._active_index]
            self._bbox = annotation.bbox
            self._current_class_id = annotation.class_id
        self.bbox_changed.emit(self._bbox)
        self.active_annotation_changed.emit(self._active_index)
        self.update()

    def cycle_active_annotation(self, delta: int = 1) -> None:
        if self._annotations:
            self.set_active_annotation(self._active_index + delta)

    def update_active_class(self, class_id: int) -> None:
        self._current_class_id = class_id
        if self._multi_mode and 0 <= self._active_index < len(self._annotations):
            annotation = self._annotations[self._active_index]
            self._annotations[self._active_index] = YoloAnnotation(
                class_id=class_id,
                bbox=annotation.bbox,
            )
            self.annotations_changed.emit(tuple(self._annotations))
        self.update()

    def remove_active_annotation(self) -> bool:
        if not self._multi_mode or not (
            0 <= self._active_index < len(self._annotations)
        ):
            return False
        self._annotations.pop(self._active_index)
        if self._annotations:
            self._active_index = min(
                self._active_index,
                len(self._annotations) - 1,
            )
            annotation = self._annotations[self._active_index]
            self._bbox = annotation.bbox
            self._current_class_id = annotation.class_id
        else:
            self._active_index = -1
            self._bbox = None
        self.bbox_changed.emit(self._bbox)
        self.active_annotation_changed.emit(self._active_index)
        self.annotations_changed.emit(tuple(self._annotations))
        self.update()
        return True

    def reset_annotation(self) -> None:
        self.set_bbox(None)

    def set_class_style(
        self,
        name: str,
        color: str,
        class_id: Optional[int] = None,
    ) -> None:
        self._class_name = name
        self._class_color = QColor(color)
        if class_id is not None:
            self._current_class_id = class_id
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

    def _bbox_widget_rect(self, bbox: Optional[BBox] = None) -> QRectF:
        bbox = bbox if bbox is not None else self._bbox
        if bbox is None:
            return QRectF()
        top_left = self._image_to_widget(
            QPointF(bbox.x1, bbox.y1)
        )
        bottom_right = self._image_to_widget(
            QPointF(bbox.x2, bbox.y2)
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

    def _hit_annotation(self, point: QPointF) -> int:
        for index in range(len(self._annotations) - 1, -1, -1):
            if self._bbox_widget_rect(
                self._annotations[index].bbox
            ).contains(point):
                return index
        return -1

    def _annotation_style(self, class_id: int) -> Tuple[str, QColor]:
        if 0 <= class_id < len(self._class_names):
            name = self._class_names[class_id]
        else:
            name = "Class {}".format(class_id)
        if 0 <= class_id < len(self._class_colors):
            color = QColor(self._class_colors[class_id])
        else:
            color = QColor("#38BDF8")
        return name, color

    def _paint_box(
        self,
        painter: QPainter,
        target: QRectF,
        bbox: BBox,
        label: str,
        color: QColor,
        active: bool,
    ) -> None:
        box_rect = self._bbox_widget_rect(bbox)
        pen = QPen(color, 2.8 if active else 1.8)
        pen.setCosmetic(True)
        if not active:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(32 if active else 14)
        painter.setBrush(fill)
        painter.drawRect(box_rect)

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
        label_fill = QColor(color)
        if not active:
            label_fill.setAlpha(210)
        painter.setBrush(label_fill)
        painter.drawRoundedRect(label_rect, 4, 4)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(
            label_rect.adjusted(8, 0, -8, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            label,
        )

        if active:
            painter.setPen(QPen(QColor("#FFFFFF"), 1))
            painter.setBrush(color)
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

        if self._multi_mode:
            for index, annotation in enumerate(self._annotations):
                if index == self._active_index:
                    continue
                name, color = self._annotation_style(annotation.class_id)
                self._paint_box(
                    painter,
                    target,
                    annotation.bbox,
                    name,
                    color,
                    False,
                )
            if 0 <= self._active_index < len(self._annotations):
                annotation = self._annotations[self._active_index]
                name, color = self._annotation_style(annotation.class_id)
                self._paint_box(
                    painter,
                    target,
                    annotation.bbox,
                    name,
                    color,
                    True,
                )
        elif self._bbox is not None:
            self._paint_box(
                painter,
                target,
                self._bbox,
                self._class_name or "BBox",
                self._class_color,
                True,
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
        elif self._multi_mode and self._hit_annotation(point) >= 0:
            hit_index = self._hit_annotation(point)
            if hit_index != self._active_index:
                self.set_active_annotation(hit_index)
                self._original_bbox = self._bbox
            self._mode = "move"
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
            if self._multi_mode:
                self._annotations.append(
                    YoloAnnotation(
                        class_id=self._current_class_id,
                        bbox=self._bbox,
                    )
                )
                self._active_index = len(self._annotations) - 1
                self.active_annotation_changed.emit(self._active_index)
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
        if (
            self._multi_mode
            and self._bbox is not None
            and 0 <= self._active_index < len(self._annotations)
        ):
            annotation = self._annotations[self._active_index]
            self._annotations[self._active_index] = YoloAnnotation(
                class_id=annotation.class_id,
                bbox=self._bbox,
            )
            self.annotations_changed.emit(tuple(self._annotations))
        self.bbox_changed.emit(self._bbox)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode in {"draw", "move", "resize"}:
            if self._bbox is not None and not self._bbox.is_valid():
                if (
                    self._multi_mode
                    and 0 <= self._active_index < len(self._annotations)
                ):
                    self._annotations.pop(self._active_index)
                    self._active_index = min(
                        self._active_index,
                        len(self._annotations) - 1,
                    )
                    if self._active_index >= 0:
                        annotation = self._annotations[self._active_index]
                        self._bbox = annotation.bbox
                        self._current_class_id = annotation.class_id
                    else:
                        self._bbox = None
                    self.active_annotation_changed.emit(self._active_index)
                else:
                    self._bbox = None
            if self._multi_mode:
                self.annotations_changed.emit(tuple(self._annotations))
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
