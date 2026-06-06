import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QSettings


@pytest.fixture(autouse=True)
def isolate_qsettings(tmp_path):
    settings_root = tmp_path / "qsettings"
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(
        QSettings.IniFormat,
        QSettings.UserScope,
        str(settings_root),
    )
    QCoreApplication.setOrganizationName("cvat-nhai-tests")
    QCoreApplication.setApplicationName("cvat-nhai-tests")
    settings = QSettings()
    settings.clear()
    yield
    settings.clear()
