from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage

from cvat_nhai.canvas import AnnotationCanvas


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
