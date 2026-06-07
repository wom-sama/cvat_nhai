from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter

from cvat_nhai.canvas import AnnotationCanvas
from cvat_nhai.models import BBox, YoloAnnotation


def test_canvas_draw_reset_and_zoom(qtbot) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(400, 300, QImage.Format_RGB32))
    canvas.show()
    qtbot.addWidget(canvas)
    qtbot.waitExposed(canvas)

    qtbot.mousePress(canvas, Qt.LeftButton, pos=QPoint(250, 200))
    qtbot.mouseMove(canvas, QPoint(550, 400))
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(550, 400))

    assert canvas.bbox is not None
    assert canvas.bbox.is_valid()
    canvas.reset_annotation()
    assert canvas.bbox is None


def test_canvas_keeps_image_visible_while_loading_replacement(qtbot) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(400, 300, QImage.Format_RGB32))
    canvas.show()
    qtbot.addWidget(canvas)

    canvas.set_loading("Dang tai next.jpg...")

    assert canvas.is_loading
    assert canvas.image_size == (400, 300)
    replacement = QImage(640, 480, QImage.Format_RGB32)
    canvas.set_image(replacement)
    assert not canvas.is_loading
    assert canvas.image_size == (640, 480)


def test_canvas_multiple_annotations_select_class_and_remove(qtbot) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(400, 300, QImage.Format_RGB32))
    canvas.set_class_catalog(("a", "b"), ("#00FF00", "#FF0000"))
    canvas.set_annotations(
        (
            YoloAnnotation(0, BBox(20, 20, 120, 120)),
            YoloAnnotation(1, BBox(200, 100, 350, 260)),
        ),
        active_index=1,
    )
    canvas.show()
    qtbot.addWidget(canvas)
    qtbot.waitExposed(canvas)

    assert len(canvas.annotations) == 2
    assert canvas.active_index == 1
    canvas.update_active_class(0)
    assert canvas.annotations[1].class_id == 0
    canvas.cycle_active_annotation()
    assert canvas.active_index == 0
    assert canvas.remove_active_annotation()
    assert len(canvas.annotations) == 1


def test_bbox_render_has_no_interior_overlay(qtbot) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(800, 600)
    image = QImage(400, 300, QImage.Format_RGB32)
    image.fill(QColor("#718096"))
    canvas.set_image(image)
    canvas.set_class_catalog(("mango",), ("#22C55E",))
    canvas.set_annotations(
        (YoloAnnotation(0, BBox(80, 60, 320, 240)),),
        active_index=0,
    )
    canvas.show()
    qtbot.addWidget(canvas)
    qtbot.waitExposed(canvas)

    rendered = QImage(canvas.size(), QImage.Format_ARGB32)
    rendered.fill(QColor("black"))
    painter = QPainter(rendered)
    canvas.render(painter, QPoint())
    painter.end()

    center = canvas._image_to_widget(QPoint(200, 150))
    interior_color = rendered.pixelColor(
        int(round(center.x())),
        int(round(center.y())),
    )
    assert interior_color == QColor("#718096")


def test_tiny_bbox_has_visible_click_target_without_geometry_change(
    qtbot,
) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(640, 640, QImage.Format_RGB32))
    tiny = YoloAnnotation(0, BBox(20.0, 90.0, 21.4, 91.4))
    large = YoloAnnotation(1, BBox(100, 100, 500, 500))
    canvas.set_class_catalog(("tiny", "large"), ("#22C55E", "#EF4444"))
    canvas.set_annotations((tiny, large), active_index=1)
    canvas.show()
    qtbot.addWidget(canvas)
    qtbot.waitExposed(canvas)

    target = canvas._image_to_widget(QPoint(21, 91))
    qtbot.mouseClick(
        canvas,
        Qt.LeftButton,
        pos=QPoint(round(target.x()), round(target.y())),
    )

    assert canvas.active_index == 0
    assert canvas.annotations[0].bbox == tiny.bbox
    visible = canvas._visible_bbox_widget_rect(tiny.bbox)
    assert visible.width() >= canvas.MIN_VISIBLE_BOX_SIZE
    assert visible.height() >= canvas.MIN_VISIBLE_BOX_SIZE
