import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSettings, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QImageReader, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .canvas import AnnotationCanvas
from .constants import (
    APP_NAME,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_NAMES,
    DEFAULT_CLASSIFICATION_ROOT,
    DEFAULT_DETECTION_ROOT,
)
from .dataset import DatasetError, DatasetManager
from .migration import apply_migration, plan_migration
from .models import BBox, DatasetPaths, MigrationReport
from .scanner import scan_images
from .schema import audit_schema, initialize_empty_datasets
from .settings_dialog import SettingsDialog
from .workers import FunctionTask


def load_qimage(path: Path) -> QImage:
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        raise RuntimeError(
            "Khong doc duoc anh {}: {}".format(path, reader.errorString())
        )
    return image


class MainWindow(QMainWindow):
    request_focus_canvas = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(980, 640)

        self.settings = QSettings()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(max(2, min(8, self.thread_pool.maxThreadCount())))

        project_root = Path(__file__).resolve().parents[2]
        self.detection_root = Path(
            self.settings.value(
                "datasets/detection_root",
                str(DEFAULT_DETECTION_ROOT),
            )
        )
        self.classification_root = Path(
            self.settings.value(
                "datasets/classification_root",
                str(DEFAULT_CLASSIFICATION_ROOT),
            )
        )
        self.archive_root = Path(
            self.settings.value(
                "datasets/archive_root",
                str(project_root / "work" / "archive"),
            )
        )
        self.crop_padding = float(
            self.settings.value("datasets/crop_padding", 0.08)
        )
        output_root_value = str(
            self.settings.value("datasets/output_root", "")
        ).strip()
        self.output_root = Path(output_root_value) if output_root_value else None
        self.source_root = Path(
            self.settings.value("source/root", "")
        ) if self.settings.value("source/root", "") else None

        self.manager = self._make_manager()
        self.images: List[Path] = []
        self.current_index = 0
        self.current_path: Optional[Path] = None
        self.selected_class = -1
        self.busy = False
        self.image_cache: "OrderedDict[str, QImage]" = OrderedDict()
        self.pending_loads = set()
        self.active_tasks = set()
        self.class_buttons: List[QPushButton] = []

        self._build_ui()
        self._build_actions()
        self._apply_style()
        self._update_schema_status()
        if self.source_root and self.source_root.is_dir():
            self.source_edit.setText(str(self.source_root))
            self.scan_source()

    def _make_manager(self) -> DatasetManager:
        return DatasetManager(
            DatasetPaths(
                detection_root=self.detection_root,
                classification_root=self.classification_root,
                archive_root=self.archive_root,
            ),
            crop_padding=self.crop_padding,
        )

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(330)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        subtitle = QLabel("Gan nhan nhanh - 5 lop xoai")
        subtitle.setObjectName("muted")
        side.addWidget(title)
        side.addWidget(subtitle)

        source_label = QLabel("THU MUC NGUON")
        source_label.setObjectName("sectionLabel")
        side.addWidget(source_label)
        source_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Chon thu muc anh can gan lai nhan")
        self.source_edit.returnPressed.connect(self.scan_source)
        browse = QToolButton()
        browse.setText("...")
        browse.setToolTip("Chon thu muc (Ctrl+O)")
        browse.clicked.connect(self.choose_source)
        source_row.addWidget(self.source_edit)
        source_row.addWidget(browse)
        side.addLayout(source_row)

        self.scan_button = QPushButton("Quet anh")
        self.scan_button.clicked.connect(self.scan_source)
        side.addWidget(self.scan_button)

        destination_label = QLabel("THU MUC DICH")
        destination_label.setObjectName("sectionLabel")
        side.addWidget(destination_label)
        destination_row = QHBoxLayout()
        self.destination_edit = QLineEdit()
        self.destination_edit.setPlaceholderText(
            "Tao dataset/ va cls_crops/ trong thu muc nay"
        )
        if self.output_root is not None:
            self.destination_edit.setText(str(self.output_root))
        self.destination_edit.returnPressed.connect(
            self.apply_destination_from_text
        )
        destination_browse = QToolButton()
        destination_browse.setText("...")
        destination_browse.setToolTip("Chon thu muc dich")
        destination_browse.clicked.connect(self.choose_destination)
        destination_row.addWidget(self.destination_edit)
        destination_row.addWidget(destination_browse)
        side.addLayout(destination_row)
        self.destination_hint = QLabel(
            "Tao: dataset (YOLO) + cls_crops (classification)"
        )
        self.destination_hint.setObjectName("muted")
        side.addWidget(self.destination_hint)

        progress_row = QHBoxLayout()
        self.queue_label = QLabel("0 anh")
        self.queue_label.setObjectName("muted")
        self.position_label = QLabel("0 / 0")
        self.position_label.setObjectName("muted")
        progress_row.addWidget(self.queue_label)
        progress_row.addStretch()
        progress_row.addWidget(self.position_label)
        side.addLayout(progress_row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        side.addWidget(self.progress)

        class_label = QLabel("CHON CLASS (PHIM 1-5)")
        class_label.setObjectName("sectionLabel")
        side.addWidget(class_label)
        for index, (raw_name, label, color) in enumerate(
            zip(CLASS_NAMES, CLASS_LABELS, CLASS_COLORS)
        ):
            button = QPushButton()
            button.setCheckable(True)
            button.setProperty("classButton", True)
            button.setProperty("classColor", color)
            button.setMinimumHeight(58)
            button.setText("{}   {}\n     {}".format(index + 1, label, raw_name))
            button.setToolTip("Phim {}".format(index + 1))
            button.clicked.connect(
                lambda checked=False, value=index: self.select_class(value)
            )
            self.class_buttons.append(button)
            side.addWidget(button)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Split:"))
        self.split_combo = QComboBox()
        self.split_combo.addItem("Tu dong 70/20/10", "auto")
        self.split_combo.addItem("Train", "train")
        self.split_combo.addItem("Validation", "val")
        self.split_combo.addItem("Test", "test")
        split_row.addWidget(self.split_combo, 1)
        side.addLayout(split_row)

        self.bbox_label = QLabel("BBox: chua ve")
        self.bbox_label.setObjectName("muted")
        side.addWidget(self.bbox_label)

        side.addSpacerItem(
            QSpacerItem(
                10,
                10,
                QSizePolicy.Minimum,
                QSizePolicy.Expanding,
            )
        )

        self.commit_button = QPushButton("ENTER  Luu va chuyen anh")
        self.commit_button.setObjectName("primaryButton")
        self.commit_button.setMinimumHeight(44)
        self.commit_button.clicked.connect(self.commit_current)
        side.addWidget(self.commit_button)

        action_row = QHBoxLayout()
        self.reset_button = QPushButton("F  Lam lai")
        self.reset_button.clicked.connect(self.reset_annotation)
        self.delete_button = QPushButton("DEL  Loai bo")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.clicked.connect(self.delete_current)
        action_row.addWidget(self.reset_button)
        action_row.addWidget(self.delete_button)
        side.addLayout(action_row)

        undo = QPushButton("Ctrl+Z  Hoan tac thao tac gan nhat")
        undo.clicked.connect(self.undo_latest)
        side.addWidget(undo)

        side_scroll = QScrollArea()
        side_scroll.setObjectName("sideScroll")
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setFixedWidth(342)
        side_scroll.setWidget(sidebar)
        outer.addWidget(side_scroll)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(18, 14, 18, 14)
        content_layout.setSpacing(10)

        top = QHBoxLayout()
        file_block = QVBoxLayout()
        self.file_label = QLabel("Chua co anh")
        self.file_label.setObjectName("fileTitle")
        self.path_label = QLabel("")
        self.path_label.setObjectName("muted")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        file_block.addWidget(self.file_label)
        file_block.addWidget(self.path_label)
        top.addLayout(file_block, 1)

        self.schema_badge = QLabel()
        self.schema_badge.setObjectName("schemaBadge")
        top.addWidget(self.schema_badge)
        self.migrate_button = QPushButton("Chuyen dataset sang 5 lop")
        self.migrate_button.clicked.connect(self.start_migration)
        top.addWidget(self.migrate_button)
        settings_button = QPushButton("Cau hinh")
        settings_button.clicked.connect(self.open_settings)
        top.addWidget(settings_button)
        content_layout.addLayout(top)

        self.canvas = AnnotationCanvas()
        self.canvas.bbox_changed.connect(self._bbox_changed)
        content_layout.addWidget(self.canvas, 1)

        help_row = QHBoxLayout()
        help_text = QLabel(
            "Keo chuot: tao bbox   |   Keo trong khung: di chuyen   |   "
            "Keo diem vuong: resize   |   Wheel: zoom   |   Space+drag: pan   |   A/D: anh truoc/sau"
        )
        help_text.setObjectName("muted")
        help_row.addWidget(help_text)
        help_row.addStretch()
        fit_button = QPushButton("Fit anh")
        fit_button.clicked.connect(self.canvas.fit_to_view)
        help_row.addWidget(fit_button)
        content_layout.addLayout(help_row)

        outer.addWidget(content, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage("San sang")

    def _build_actions(self) -> None:
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.choose_source)
        self.addAction(open_action)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111827;
                color: #E5E7EB;
                font-family: "Segoe UI";
                font-size: 10pt;
            }
            #sidebar { background: #0F172A; border-right: 1px solid #263246; }
            #sideScroll { background: #0F172A; border: 0; }
            #appTitle { font-size: 22pt; font-weight: 700; color: #F8FAFC; }
            #fileTitle { font-size: 14pt; font-weight: 600; color: #F8FAFC; }
            #muted { color: #94A3B8; }
            #sectionLabel {
                color: #64748B;
                font-size: 8pt;
                font-weight: 700;
                margin-top: 8px;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                background: #182235;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 8px;
                color: #F8FAFC;
            }
            QPushButton, QToolButton {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 8px 11px;
                color: #E2E8F0;
            }
            QPushButton:hover, QToolButton:hover {
                background: #26364D;
                border-color: #475569;
            }
            QPushButton:disabled { color: #64748B; background: #172033; }
            QPushButton[classButton="true"] {
                text-align: left;
                padding: 8px 12px;
                font-size: 9pt;
            }
            QPushButton[classButton="true"]:checked {
                background: #1D3244;
                border: 2px solid #38BDF8;
                color: #FFFFFF;
            }
            #primaryButton {
                background: #0284C7;
                border-color: #0EA5E9;
                color: white;
                font-weight: 700;
            }
            #primaryButton:hover { background: #0369A1; }
            #dangerButton { color: #FCA5A5; }
            #dangerButton:hover { background: #451A1A; border-color: #7F1D1D; }
            #schemaBadge {
                border-radius: 10px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QProgressBar { background: #1E293B; border: 0; border-radius: 3px; }
            QProgressBar::chunk { background: #0EA5E9; border-radius: 3px; }
            QStatusBar { background: #0F172A; color: #94A3B8; }
            """
        )

    def choose_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chon thu muc anh nguon",
            self.source_edit.text() or str(Path.home()),
        )
        if folder:
            self.source_edit.setText(folder)
            self.scan_source()

    def choose_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chon thu muc dich",
            self.destination_edit.text() or str(Path.home()),
        )
        if folder:
            self.destination_edit.setText(folder)
            self.apply_destination_from_text()

    def apply_destination_from_text(self) -> None:
        text = self.destination_edit.text().strip()
        if not text:
            self.statusBar().showMessage("Chon thu muc dich", 3000)
            return
        try:
            self.set_destination_root(Path(text))
        except Exception as error:
            self._show_error(str(error))

    def set_destination_root(self, root: Path) -> None:
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        detection_root = root / "dataset"
        classification_root = root / "cls_crops"
        archive_root = root / ".cvat_nhai_archive"
        initialize_empty_datasets(detection_root, classification_root)

        self.output_root = root
        self.detection_root = detection_root
        self.classification_root = classification_root
        self.archive_root = archive_root
        self.destination_edit.setText(str(root))
        self.settings.setValue("datasets/output_root", str(root))
        self.settings.setValue(
            "datasets/detection_root",
            str(self.detection_root),
        )
        self.settings.setValue(
            "datasets/classification_root",
            str(self.classification_root),
        )
        self.settings.setValue(
            "datasets/archive_root",
            str(self.archive_root),
        )
        self.manager = self._make_manager()
        self._update_schema_status()
        self.statusBar().showMessage(
            "Da chon dich: {} (dataset + cls_crops)".format(root),
            5000,
        )
        if self.source_root and self.source_root.is_dir():
            self.scan_source()

    def scan_source(self) -> None:
        text = self.source_edit.text().strip()
        if not text:
            return
        root = Path(text)
        if not root.is_dir():
            self._show_error("Thu muc khong ton tai: {}".format(root))
            return
        self.source_root = root.resolve()
        self.settings.setValue("source/root", str(self.source_root))
        self.scan_button.setEnabled(False)
        self.scan_button.setText("Dang quet...")
        self.statusBar().showMessage("Dang quet thu muc va cac thu muc con...")
        task = FunctionTask(
            scan_images,
            self.source_root,
            (
                self.detection_root,
                self.classification_root,
                self.archive_root,
            ),
        )
        task.signals.succeeded.connect(self._scan_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda: (
                self.scan_button.setEnabled(True),
                self.scan_button.setText("Quet anh"),
            )
        )
        self._start_task(task)

    def _start_task(self, task: FunctionTask) -> None:
        """Keep the Python QRunnable wrapper alive until queued signals finish."""
        self.active_tasks.add(task)
        task.signals.finished.connect(
            lambda value=task: self.active_tasks.discard(value)
        )
        self.thread_pool.start(task)

    def _scan_finished(self, result: object) -> None:
        self.images = list(result)  # type: ignore[arg-type]
        self.current_index = 0
        self.image_cache.clear()
        self.queue_label.setText("{} anh".format(len(self.images)))
        self.progress.setMaximum(max(1, len(self.images)))
        self.progress.setValue(0)
        self.statusBar().showMessage(
            "Da tim thay {} anh".format(len(self.images)),
            4000,
        )
        self.show_current()

    def show_current(self) -> None:
        while self.images and not self.images[self.current_index].exists():
            self.images.pop(self.current_index)
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)
        if not self.images:
            self.current_path = None
            self.canvas.clear_image()
            self.file_label.setText("Khong con anh")
            self.path_label.clear()
            self.position_label.setText("0 / 0")
            self.queue_label.setText("0 anh")
            self.progress.setValue(self.progress.maximum())
            return

        self.current_index = max(0, min(self.current_index, len(self.images) - 1))
        self.current_path = self.images[self.current_index]
        self.file_label.setText(self.current_path.name)
        self.path_label.setText(str(self.current_path))
        self.position_label.setText(
            "{} / {}".format(self.current_index + 1, len(self.images))
        )
        self.queue_label.setText("{} anh con lai".format(len(self.images)))
        self.progress.setMaximum(max(1, len(self.images)))
        self.progress.setValue(self.current_index)
        self.canvas.reset_annotation()
        self._load_image(self.current_path, display=True)
        if self.current_index + 1 < len(self.images):
            self._load_image(self.images[self.current_index + 1], display=False)

    def _load_image(self, path: Path, display: bool) -> None:
        key = str(path)
        cached = self.image_cache.get(key)
        if cached is not None:
            self.image_cache.move_to_end(key)
            if display and self.current_path == path:
                self.canvas.set_image(cached)
                self._bbox_changed(None)
            return
        if key in self.pending_loads:
            return
        self.pending_loads.add(key)
        if display:
            self.canvas.clear_image()
            self.statusBar().showMessage("Dang tai {}...".format(path.name))
        task = FunctionTask(load_qimage, path)
        task.signals.succeeded.connect(
            lambda image, value=path, show=display: self._image_loaded(
                value,
                image,
                show,
            )
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda value=key: self.pending_loads.discard(value)
        )
        self._start_task(task)

    def _image_loaded(self, path: Path, result: object, display: bool) -> None:
        image = result
        if not isinstance(image, QImage):
            return
        key = str(path)
        self.image_cache[key] = image
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > 8:
            self.image_cache.popitem(last=False)
        if display and self.current_path == path:
            self.canvas.set_image(image)
            self.statusBar().showMessage(
                "{} x {} px".format(image.width(), image.height()),
                3000,
            )

    def select_class(self, class_id: int) -> None:
        self.selected_class = class_id
        for index, button in enumerate(self.class_buttons):
            button.setChecked(index == class_id)
        self.canvas.set_class_style(
            CLASS_LABELS[class_id],
            CLASS_COLORS[class_id],
        )
        self.statusBar().showMessage(
            "Class {}: {}".format(class_id, CLASS_NAMES[class_id]),
            2500,
        )
        self.canvas.setFocus()

    def reset_annotation(self) -> None:
        if self.busy:
            return
        self.selected_class = -1
        for button in self.class_buttons:
            button.setChecked(False)
        self.canvas.set_class_style("", "#22C55E")
        self.canvas.reset_annotation()
        self.statusBar().showMessage("Da reset bbox va class", 2500)

    def _bbox_changed(self, value: object) -> None:
        if isinstance(value, BBox):
            self.bbox_label.setText(
                "BBox: {:.0f} x {:.0f} px".format(value.width, value.height)
            )
        else:
            self.bbox_label.setText("BBox: chua ve")

    def commit_current(self) -> None:
        if self.busy or self.current_path is None:
            return
        if self.selected_class < 0:
            self.statusBar().showMessage("Chon class bang phim 1-5", 3500)
            return
        bbox = self.canvas.bbox
        if bbox is None or not bbox.is_valid():
            self.statusBar().showMessage("Keo mot bbox hop le tren anh", 3500)
            return
        audit = audit_schema(self.detection_root, self.classification_root)
        if not audit.ready:
            self.statusBar().showMessage(
                "Dataset chua san sang: can chuyen sang schema 5 lop",
                5000,
            )
            return

        path = self.current_path
        class_id = self.selected_class
        requested_split = str(self.split_combo.currentData())
        self._set_busy(True, "Dang ghi hai dataset...")
        task = FunctionTask(
            self.manager.annotate,
            path,
            bbox,
            class_id,
            requested_split,
        )
        task.signals.succeeded.connect(
            lambda result, value=path: self._operation_finished(value, result)
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def delete_current(self) -> None:
        if self.busy or self.current_path is None:
            return
        path = self.current_path
        self._set_busy(True, "Dang dua anh vao safety archive...")
        task = FunctionTask(self.manager.reject, path)
        task.signals.succeeded.connect(
            lambda result, value=path: self._operation_finished(value, result)
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _operation_finished(self, path: Path, result: object) -> None:
        if path in self.images:
            index = self.images.index(path)
            self.images.pop(index)
            if index < self.current_index:
                self.current_index -= 1
            self.current_index = min(self.current_index, max(0, len(self.images) - 1))
        self.image_cache.pop(str(path), None)
        action = getattr(result, "action", "")
        if action == "annotate":
            split = getattr(result, "split", "")
            self.statusBar().showMessage(
                "Da luu detection + crop vao split {}".format(split),
                4500,
            )
        else:
            self.statusBar().showMessage(
                "Da loai anh; Ctrl+Z de phuc hoi",
                4500,
            )
        self.show_current()

    def undo_latest(self) -> None:
        if self.busy:
            return
        self._set_busy(True, "Dang hoan tac...")
        task = FunctionTask(self.manager.undo_latest)
        task.signals.succeeded.connect(self._undo_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _undo_finished(self, result: object) -> None:
        if isinstance(result, Path):
            if result not in self.images:
                self.images.insert(min(self.current_index, len(self.images)), result)
            self.statusBar().showMessage("Da hoan tac va phuc hoi anh", 4000)
            self.show_current()
        else:
            self.statusBar().showMessage("Khong co thao tac de hoan tac", 3000)

    def navigate(self, delta: int) -> None:
        if self.busy or not self.images:
            return
        new_index = self.current_index + delta
        if 0 <= new_index < len(self.images):
            self.current_index = new_index
            self.show_current()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        self.commit_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message)

    def _update_schema_status(self) -> None:
        try:
            audit = audit_schema(
                self.detection_root,
                self.classification_root,
            )
        except Exception as error:
            self.schema_badge.setText("Loi cau hinh")
            self.schema_badge.setStyleSheet(
                "background:#451A1A;color:#FCA5A5;"
            )
            self.migrate_button.setVisible(True)
            self.statusBar().showMessage(str(error), 5000)
            return
        if audit.ready:
            self.schema_badge.setText("5 lop - san sang")
            self.schema_badge.setStyleSheet(
                "background:#123524;color:#86EFAC;"
            )
            self.migrate_button.setVisible(False)
        else:
            self.schema_badge.setText("Can migration 4 -> 5 lop")
            self.schema_badge.setStyleSheet(
                "background:#422006;color:#FCD34D;"
            )
            self.migrate_button.setVisible(True)

    def start_migration(self) -> None:
        if self.busy:
            return
        answer = QMessageBox.question(
            self,
            "Chuyen detection sang 5 lop",
            "Tool se doi ID label detection dua tren cls_crops/manifest.csv.\n"
            "Mot file ZIP backup se duoc tao truoc khi ghi.\n\n"
            "Tiep tuc kiem tra va migration?",
        )
        if answer != QMessageBox.Yes:
            return
        self._set_busy(True, "Dang kiem tra toan bo label va manifest...")
        task = FunctionTask(
            apply_migration,
            self.detection_root,
            self.classification_root,
        )
        task.signals.succeeded.connect(self._migration_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _migration_finished(self, result: object) -> None:
        if not isinstance(result, MigrationReport):
            return
        self._update_schema_status()
        QMessageBox.information(
            self,
            "Migration hoan tat",
            "Da chuyen {} file label / {} object sang schema 5 lop.\n"
            "Backup: {}".format(
                result.changed_labels,
                result.total_objects,
                result.backup_path,
            ),
        )

    def open_settings(self) -> None:
        dialog = SettingsDialog(
            self.detection_root,
            self.classification_root,
            self.archive_root,
            self.crop_padding,
            self,
        )
        if dialog.exec() != dialog.Accepted:
            return
        values = dialog.values()
        self.detection_root = values["detection_root"]  # type: ignore[assignment]
        self.classification_root = values["classification_root"]  # type: ignore[assignment]
        self.archive_root = values["archive_root"]  # type: ignore[assignment]
        self.crop_padding = float(values["crop_padding"])
        self.settings.setValue(
            "datasets/detection_root",
            str(self.detection_root),
        )
        self.settings.setValue(
            "datasets/classification_root",
            str(self.classification_root),
        )
        self.settings.setValue(
            "datasets/archive_root",
            str(self.archive_root),
        )
        self.settings.setValue(
            "datasets/crop_padding",
            self.crop_padding,
        )
        self.output_root = None
        self.settings.setValue("datasets/output_root", "")
        self.destination_edit.clear()
        self.manager = self._make_manager()
        self._update_schema_status()

    def _background_failed(self, traceback_text: str) -> None:
        lines = traceback_text.strip().splitlines()
        message = lines[-1] if lines else "Loi khong xac dinh"
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 7000)
        QMessageBox.critical(self, "Loi", message)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.settings.sync()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_latest()
            event.accept()
            return
        if Qt.Key_1 <= key <= Qt.Key_5:
            self.select_class(key - Qt.Key_1)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.commit_current()
            event.accept()
            return
        if key == Qt.Key_Delete:
            self.delete_current()
            event.accept()
            return
        if key == Qt.Key_F:
            self.reset_annotation()
            event.accept()
            return
        if key == Qt.Key_A:
            self.navigate(-1)
            event.accept()
            return
        if key == Qt.Key_D:
            self.navigate(1)
            event.accept()
            return
        super().keyPressEvent(event)
