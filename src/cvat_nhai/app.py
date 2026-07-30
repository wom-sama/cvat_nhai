import sys

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from .constants import APP_NAME, ORG_NAME
from .main_window import MainWindow


def _application_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#0284C7"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 14, 14)
    painter.setBrush(QColor("#F8FAFC"))
    painter.drawRect(15, 17, 34, 4)
    painter.drawRect(15, 43, 34, 4)
    painter.drawRect(15, 17, 4, 30)
    painter.drawRect(45, 17, 4, 30)
    painter.end()
    return QIcon(pixmap)


def _load_ui_font() -> None:
    # Explicit registration also fixes missing system fonts in Qt offscreen/RDP sessions.
    QFontDatabase.addApplicationFont(r"C:\Windows\Fonts\segoeui.ttf")


def main() -> int:
    QCoreApplication.setOrganizationName(ORG_NAME)
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion("1.5.0")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    _load_ui_font()
    app.setStyle("Fusion")
    app.setWindowIcon(_application_icon())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
