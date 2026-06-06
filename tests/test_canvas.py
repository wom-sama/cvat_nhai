from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage

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
