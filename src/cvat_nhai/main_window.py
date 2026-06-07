from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QEvent, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QImageReader, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
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
from .export_progress_dialog import ExportProgressDialog
from .migration import apply_migration
from .models import BBox, DatasetPaths, MigrationReport
from .scanner import scan_images
from .schema import audit_schema, initialize_empty_datasets
from .seek_slider import SeekSlider
from .settings_dialog import SettingsDialog
from .workers import FunctionTask
from .yolo_editor import (
    YoloDatasetEditor,
    YoloEditorError,
    export_rebalanced_datasets,
    read_yolo_annotations,
    scan_yolo_dataset,
)


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
    NAVIGATION_INITIAL_DELAY_MS = 260
    NAVIGATION_REPEAT_MS = 140

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
        self.mode = "label"
        self.active_class_names = tuple(CLASS_NAMES)
        self.editor_index = None
        self.editor_manager = None
        self.editor_samples_by_path = {}
        self.editor_original_annotations = ()
        self.images: List[Path] = []
        self.current_index = 0
        self.current_path: Optional[Path] = None
        self.selected_class = -1
        self.busy = False
        self.image_cache: "OrderedDict[str, QImage]" = OrderedDict()
        self.pending_loads = set()
        self.active_tasks = set()
        self.class_buttons: List[QPushButton] = []
        self.navigation_direction = 0
        self.navigation_timer = QTimer(self)
        self.navigation_timer.setSingleShot(False)
        self.navigation_timer.timeout.connect(
            self._repeat_navigation
        )
        self.export_task: Optional[FunctionTask] = None
        self.export_progress_dialog: Optional[
            ExportProgressDialog
        ] = None

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

        mode_label = QLabel("CHE DO")
        mode_label.setObjectName("sectionLabel")
        side.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Gan nhan anh moi", "label")
        self.mode_combo.addItem("Xem / sua dataset YOLO cu", "edit")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        side.addWidget(self.mode_combo)

        self.source_label = QLabel("THU MUC NGUON")
        self.source_label.setObjectName("sectionLabel")
        side.addWidget(self.source_label)
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

        self.destination_label = QLabel("THU MUC DICH")
        self.destination_label.setObjectName("sectionLabel")
        side.addWidget(self.destination_label)
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
        self.destination_browse = QToolButton()
        self.destination_browse.setText("...")
        self.destination_browse.setToolTip("Chon thu muc dich")
        self.destination_browse.clicked.connect(self.choose_destination)
        destination_row.addWidget(self.destination_edit)
        destination_row.addWidget(self.destination_browse)
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
        self.progress = SeekSlider()
        self.progress.setRange(0, 0)
        self.progress.setEnabled(False)
        self.progress.setFixedHeight(18)
        self.progress.seek_requested.connect(self.seek_to_index)
        self.progress.valueChanged.connect(self._preview_seek_position)
        side.addWidget(self.progress)

        self.export_button = QPushButton(
            "Xuat yolo_f + class_f"
        )
        self.export_button.clicked.connect(self.export_editor_dataset)
        self.export_button.setVisible(False)
        side.addWidget(self.export_button)

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
        self.split_label = QLabel("Split:")
        split_row.addWidget(self.split_label)
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

        self.undo_button = QPushButton("Ctrl+Z  Hoan tac thao tac gan nhat")
        self.undo_button.clicked.connect(self.undo_latest)
        side.addWidget(self.undo_button)

        self.remove_box_button = QPushButton("Backspace  Xoa box dang chon")
        self.remove_box_button.clicked.connect(self.remove_active_box)
        self.remove_box_button.setVisible(False)
        side.addWidget(self.remove_box_button)

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
        self.canvas.set_class_catalog(CLASS_NAMES, CLASS_COLORS)
        self.canvas.bbox_changed.connect(self._bbox_changed)
        self.canvas.active_annotation_changed.connect(
            self._active_annotation_changed
        )
        content_layout.addWidget(self.canvas, 1)

        help_row = QHBoxLayout()
        self.help_text = QLabel(
            "Keo chuot: tao bbox   |   Keo trong khung: di chuyen   |   "
            "Keo diem vuong: resize   |   Wheel: zoom   |   Space+drag: pan   |   Giu A/D: anh truoc/sau"
        )
        self.help_text.setObjectName("muted")
        help_row.addWidget(self.help_text)
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

        self.next_box_action = QAction(self)
        self.next_box_action.setShortcut(QKeySequence("Tab"))
        self.next_box_action.setShortcutContext(Qt.ApplicationShortcut)
        self.next_box_action.setEnabled(False)
        self.next_box_action.triggered.connect(
            lambda: self._cycle_editor_box(1)
        )
        self.addAction(self.next_box_action)

        self.previous_box_action = QAction(self)
        self.previous_box_action.setShortcut(QKeySequence("Shift+Tab"))
        self.previous_box_action.setShortcutContext(Qt.ApplicationShortcut)
        self.previous_box_action.setEnabled(False)
        self.previous_box_action.triggered.connect(
            lambda: self._cycle_editor_box(-1)
        )
        self.addAction(self.previous_box_action)

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
            QSlider::groove:horizontal {
                background: #1E293B;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0EA5E9;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #1E293B;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #E2E8F0;
                border: 1px solid #0EA5E9;
                width: 13px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider:disabled::handle:horizontal {
                background: transparent;
                border: 0;
                width: 1px;
            }
            QStatusBar { background: #0F172A; color: #94A3B8; }
            """
        )

    def choose_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            (
                "Chon dataset YOLO co data.yaml"
                if self.mode == "edit"
                else "Chon thu muc anh nguon"
            ),
            self.source_edit.text() or str(Path.home()),
        )
        if folder:
            self.source_edit.setText(folder)
            self.scan_source()

    def _mode_changed(self) -> None:
        mode = str(self.mode_combo.currentData())
        if mode == self.mode:
            return
        self.mode = mode
        self.images = []
        self.current_index = 0
        self.current_path = None
        self.editor_index = None
        self.editor_manager = None
        self.editor_samples_by_path = {}
        self.editor_original_annotations = ()
        self.image_cache.clear()
        self.canvas.clear_image()
        self.file_label.setText("Chua co anh")
        self.path_label.clear()
        self.queue_label.setText("0 anh")
        self.position_label.setText("0 / 0")
        self.progress.setValue(0)
        self.source_edit.clear()

        editing = mode == "edit"
        self.source_label.setText(
            "DATASET YOLO CU" if editing else "THU MUC NGUON"
        )
        self.source_edit.setPlaceholderText(
            (
                "Chon thu muc chua data.yaml"
                if editing
                else "Chon thu muc anh can gan lai nhan"
            )
        )
        self.scan_button.setText(
            "Mo dataset YOLO" if editing else "Quet anh"
        )
        self.destination_label.setVisible(not editing)
        self.destination_edit.setVisible(not editing)
        self.destination_browse.setVisible(not editing)
        self.destination_hint.setVisible(not editing)
        self.split_label.setVisible(not editing)
        self.split_combo.setVisible(not editing)
        self.undo_button.setVisible(not editing)
        self.remove_box_button.setVisible(editing)
        self.export_button.setVisible(editing)
        self.next_box_action.setEnabled(editing)
        self.previous_box_action.setEnabled(editing)
        self.progress.setEnabled(editing)
        self.commit_button.setText(
            (
                "ENTER  Ap dung thay doi"
                if editing
                else "ENTER  Luu va chuyen anh"
            )
        )
        self.reset_button.setText(
            "F  Ve nhan goc" if editing else "F  Lam lai"
        )
        self.delete_button.setText(
            "DEL  Xoa anh + nhan" if editing else "DEL  Loai bo"
        )
        self.help_text.setText(
            (
                "Click box: chon object   |   Keo/resize: sua box   |   "
                "Tab/Shift+Tab: doi box   |   Keo vung trong: them box   |   "
                "Backspace: xoa box   |   Giu A/D: anh truoc/sau"
                if editing
                else "Keo chuot: tao bbox   |   Keo trong khung: di chuyen   |   "
                "Keo diem vuong: resize   |   Wheel: zoom   |   Space+drag: pan   |   Giu A/D: anh truoc/sau"
            )
        )
        self._configure_class_buttons(CLASS_NAMES)
        self._update_schema_status()

    def _configure_class_buttons(self, names) -> None:
        if len(names) > len(self.class_buttons):
            raise YoloEditorError(
                "UI hien tai ho tro toi da {} class".format(
                    len(self.class_buttons)
                )
            )
        self.active_class_names = tuple(str(name) for name in names)
        for index, button in enumerate(self.class_buttons):
            visible = index < len(self.active_class_names)
            button.setVisible(visible)
            button.setChecked(False)
            if visible:
                name = self.active_class_names[index]
                label = (
                    CLASS_LABELS[index]
                    if tuple(names) == CLASS_NAMES
                    else name
                )
                button.setText(
                    "{}   {}\n     {}".format(index + 1, label, name)
                )
        self.canvas.set_class_catalog(
            self.active_class_names,
            CLASS_COLORS[: len(self.active_class_names)],
        )
        self.selected_class = -1

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
        if self.mode == "edit":
            self._scan_editor_dataset(root)
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

    def _scan_editor_dataset(self, root: Path) -> None:
        self.scan_button.setEnabled(False)
        self.scan_button.setText("Dang mo dataset...")
        self.statusBar().showMessage(
            "Dang doc data.yaml va ghep cap images/labels..."
        )
        task = FunctionTask(scan_yolo_dataset, root)
        task.signals.succeeded.connect(self._editor_scan_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda: (
                self.scan_button.setEnabled(True),
                self.scan_button.setText("Mo dataset YOLO"),
            )
        )
        self._start_task(task)

    def _editor_scan_finished(self, result: object) -> None:
        if len(result.class_names) > len(self.class_buttons):
            self._show_error(
                "Dataset co {} class, UI hien tai ho tro toi da {}".format(
                    len(result.class_names),
                    len(self.class_buttons),
                )
            )
            return
        self.editor_index = result
        self.editor_manager = YoloDatasetEditor(result)
        self.editor_samples_by_path = {
            sample.image_path: sample for sample in result.samples
        }
        self._configure_class_buttons(result.class_names)
        self.images = [sample.image_path for sample in result.samples]
        self.current_index = 0
        self.image_cache.clear()
        self.queue_label.setText("{} anh dataset".format(len(self.images)))
        self.progress.setRange(0, max(0, len(self.images) - 1))
        self.progress.setValue(0)
        self.schema_badge.setText(
            "{} lop - sua YOLO".format(len(result.class_names))
        )
        self.schema_badge.setStyleSheet(
            "background:#172554;color:#93C5FD;"
        )
        self.migrate_button.setVisible(False)
        self.statusBar().showMessage(
            "Da mo {} anh tu {}".format(len(self.images), result.data_yaml),
            5000,
        )
        self.show_current()

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
        self.progress.setRange(0, max(0, len(self.images) - 1))
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
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            return

        self.current_index = max(0, min(self.current_index, len(self.images) - 1))
        self.current_path = self.images[self.current_index]
        self.file_label.setText(self.current_path.name)
        self.path_label.setText(str(self.current_path))
        self.position_label.setText(
            "{} / {}".format(self.current_index + 1, len(self.images))
        )
        self.queue_label.setText(
            (
                "{} anh dataset".format(len(self.images))
                if self.mode == "edit"
                else "{} anh con lai".format(len(self.images))
            )
        )
        self.progress.setRange(0, max(0, len(self.images) - 1))
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
                if self.mode == "edit":
                    self._load_editor_annotations(path, cached)
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
            if self.mode == "edit":
                self._load_editor_annotations(path, image)
            self.statusBar().showMessage(
                "{} x {} px".format(image.width(), image.height()),
                3000,
            )

    def _load_editor_annotations(self, path: Path, image: QImage) -> None:
        sample = self.editor_samples_by_path.get(path)
        if sample is None or self.editor_index is None:
            return
        try:
            annotations = tuple(
                read_yolo_annotations(
                    sample.label_path,
                    image.width(),
                    image.height(),
                    len(self.editor_index.class_names),
                )
            )
        except Exception as error:
            self.canvas.set_annotations((), 0)
            self.editor_original_annotations = ()
            self._show_error(str(error))
            return
        self.editor_original_annotations = annotations
        active_index = (
            max(
                range(len(annotations)),
                key=lambda index: annotations[index].bbox.area,
            )
            if annotations
            else 0
        )
        self.canvas.set_annotations(annotations, active_index)
        if annotations:
            self._select_class_ui(annotations[active_index].class_id)
        else:
            self._select_class_ui(0)
        self._update_object_status()

    def select_class(self, class_id: int) -> None:
        if not 0 <= class_id < len(self.active_class_names):
            return
        self._select_class_ui(class_id)
        if self.mode == "edit":
            self.canvas.update_active_class(class_id)
            self._update_object_status()

    def _select_class_ui(self, class_id: int) -> None:
        self.selected_class = class_id
        for index, button in enumerate(self.class_buttons):
            button.setChecked(index == class_id)
        name = self.active_class_names[class_id]
        label = (
            CLASS_LABELS[class_id]
            if self.active_class_names == CLASS_NAMES
            else name
        )
        self.canvas.set_class_style(
            label,
            CLASS_COLORS[class_id],
            class_id,
        )
        self.statusBar().showMessage(
            "Class {}: {}".format(class_id, name),
            2500,
        )
        self.canvas.setFocus()

    def _active_annotation_changed(self, index: int) -> None:
        if self.mode != "edit":
            return
        annotations = self.canvas.annotations
        if 0 <= index < len(annotations):
            self._select_class_ui(annotations[index].class_id)
        self._update_object_status()

    def _cycle_editor_box(self, delta: int) -> None:
        if (
            self.mode != "edit"
            or self.busy
            or not self.canvas.annotations
        ):
            return
        self.canvas.cycle_active_annotation(delta)
        self.canvas.setFocus()
        self.statusBar().showMessage(
            "Dang chon object {} / {}".format(
                self.canvas.active_index + 1,
                len(self.canvas.annotations),
            ),
            2000,
        )

    def _update_object_status(self) -> None:
        if self.mode != "edit":
            return
        count = len(self.canvas.annotations)
        active = self.canvas.active_index
        self.bbox_label.setText(
            (
                "Object: {} / {}{} | Backspace xoa box".format(
                    active + 1,
                    count,
                    " | CHUA LUU" if self._editor_is_dirty() else "",
                )
                if active >= 0
                else "Object: 0 / {}{} | Keo de them box".format(
                    count,
                    " | CHUA LUU" if self._editor_is_dirty() else "",
                )
            )
        )

    def _editor_is_dirty(self) -> bool:
        return (
            self.mode == "edit"
            and tuple(self.canvas.annotations)
            != tuple(self.editor_original_annotations)
        )

    def reset_annotation(self) -> None:
        if self.busy:
            return
        if self.mode == "edit":
            active_index = (
                max(
                    range(len(self.editor_original_annotations)),
                    key=lambda index: self.editor_original_annotations[
                        index
                    ].bbox.area,
                )
                if self.editor_original_annotations
                else 0
            )
            self.canvas.set_annotations(
                self.editor_original_annotations,
                active_index,
            )
            if self.editor_original_annotations:
                self._select_class_ui(
                    self.editor_original_annotations[active_index].class_id
                )
            self._update_object_status()
            self.statusBar().showMessage(
                "Da khoi phuc nhan goc cua anh hien tai",
                3000,
            )
            return
        self.selected_class = -1
        for button in self.class_buttons:
            button.setChecked(False)
        self.canvas.set_class_style("", "#22C55E")
        self.canvas.reset_annotation()
        self.statusBar().showMessage("Da reset bbox va class", 2500)

    def _bbox_changed(self, value: object) -> None:
        if self.mode == "edit":
            self._update_object_status()
            return
        if isinstance(value, BBox):
            self.bbox_label.setText(
                "BBox: {:.0f} x {:.0f} px".format(value.width, value.height)
            )
        else:
            self.bbox_label.setText("BBox: chua ve")

    def commit_current(self) -> None:
        if self.busy or self.current_path is None:
            return
        if self.mode == "edit":
            self._commit_editor_current()
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

    def _commit_editor_current(self) -> None:
        if (
            self.editor_manager is None
            or self.editor_index is None
            or self.current_path is None
        ):
            return
        sample = self.editor_samples_by_path.get(self.current_path)
        if sample is None:
            return
        width, height = self.canvas.image_size
        annotations = self.canvas.annotations
        self._set_busy(True, "Dang backup va ghi label YOLO...")
        task = FunctionTask(
            self.editor_manager.save_annotations,
            sample,
            annotations,
            width,
            height,
        )
        task.signals.succeeded.connect(
            lambda _, saved=annotations: self._editor_save_finished(saved)
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _editor_save_finished(self, annotations) -> None:
        self.editor_original_annotations = tuple(annotations)
        self.statusBar().showMessage(
            "Da ap dung {} object vao label".format(len(annotations)),
            4000,
        )
        if self.current_index + 1 < len(self.images):
            self.current_index += 1
            self.show_current()

    def delete_current(self) -> None:
        if self.busy or self.current_path is None:
            return
        if self.mode == "edit":
            self._delete_editor_current()
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

    def _delete_editor_current(self) -> None:
        if self.editor_manager is None or self.current_path is None:
            return
        sample = self.editor_samples_by_path.get(self.current_path)
        if sample is None:
            return
        width, height = self.canvas.image_size
        path = self.current_path
        self._set_busy(True, "Dang archive anh va label...")
        task = FunctionTask(
            self.editor_manager.delete_sample,
            sample,
            width,
            height,
        )
        task.signals.succeeded.connect(
            lambda _, value=path: self._editor_delete_finished(value)
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _editor_delete_finished(self, path: Path) -> None:
        self.editor_samples_by_path.pop(path, None)
        if path in self.images:
            self.images.remove(path)
        self.image_cache.pop(str(path), None)
        self.current_index = min(
            self.current_index,
            max(0, len(self.images) - 1),
        )
        self.statusBar().showMessage(
            "Da xoa anh + label vao .cvat_nhai_editor_archive",
            5000,
        )
        self.show_current()

    def remove_active_box(self) -> None:
        if self.mode != "edit" or self.busy:
            return
        if self.canvas.remove_active_annotation():
            self._update_object_status()
            self.statusBar().showMessage(
                "Da xoa box trong bo nho; Enter de ap dung, F de phuc hoi",
                4000,
            )

    def export_editor_dataset(self) -> None:
        if self.mode != "edit" or self.editor_index is None or self.busy:
            return
        if self._editor_is_dirty():
            self.statusBar().showMessage(
                "Anh hien tai chua luu: nhan Enter hoac F truoc khi export",
                5000,
            )
            return
        destination = QFileDialog.getExistingDirectory(
            self,
            "Chon thu muc rong de tao yolo_f + class_f",
            str(self.editor_index.root.parent),
        )
        if not destination:
            return
        path = Path(destination)
        if any(path.iterdir()):
            self._show_error("Thu muc export phai rong")
            return
        self._set_busy(
            True,
            "Dang chia lai va xuat yolo_f + class_f...",
        )
        task = FunctionTask(
            export_rebalanced_datasets,
            self.editor_index,
            path,
            self.crop_padding,
            report_progress=True,
        )
        self.export_task = task
        dialog = ExportProgressDialog(self)
        self.export_progress_dialog = dialog
        dialog.cancel_requested.connect(task.cancel)
        task.signals.progress.connect(dialog.update_progress)
        task.signals.succeeded.connect(self._export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.finished.connect(self._export_task_finished)
        dialog.show()
        QTimer.singleShot(
            0,
            lambda value=task: self._start_task(value),
        )

    def _export_finished(self, result: object) -> None:
        if self.export_progress_dialog is not None:
            self.export_progress_dialog.finish()
        QMessageBox.information(
            self,
            "Export hoan tat",
            (
                "Da xuat {} crop va {} anh YOLO.\n"
                "YOLO: {}\nClassification: {}"
            ).format(
                result.objects,
                result.images,
                result.destination / "yolo_f",
                result.destination / "class_f",
            ),
        )

    def _export_failed(self, traceback_text: str) -> None:
        if self.export_progress_dialog is not None:
            self.export_progress_dialog.finish()
        lines = traceback_text.strip().splitlines()
        message = lines[-1] if lines else "Loi khong xac dinh"
        if "Export da bi huy an toan" in message:
            self.statusBar().showMessage(
                "Da huy export; thu muc tam da duoc don",
                5000,
            )
            return
        self._show_error(message)

    def _export_task_finished(self) -> None:
        if self.export_progress_dialog is not None:
            self.export_progress_dialog.finish()
        self.export_progress_dialog = None
        self.export_task = None
        self._set_busy(False)

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

    def navigate(self, delta: int) -> bool:
        if self.busy or not self.images:
            return False
        if self.mode == "edit" and self._editor_is_dirty():
            self.statusBar().showMessage(
                "Nhan Enter de luu hoac F de bo thay doi truoc khi chuyen anh",
                5000,
            )
            return False
        new_index = self.current_index + delta
        if 0 <= new_index < len(self.images):
            self.current_index = new_index
            self.show_current()
            return True
        return False

    def _start_continuous_navigation(self, direction: int) -> None:
        if direction not in (-1, 1):
            return
        if (
            self.navigation_timer.isActive()
            and self.navigation_direction == direction
        ):
            return
        self.navigation_timer.stop()
        self.navigation_direction = direction
        if not self.navigate(direction):
            self.navigation_direction = 0
            return
        self.navigation_timer.setInterval(
            self.NAVIGATION_INITIAL_DELAY_MS
        )
        self.navigation_timer.start()

    def _repeat_navigation(self) -> None:
        if not self.navigate(self.navigation_direction):
            self._stop_continuous_navigation()
            return
        if (
            self.navigation_timer.interval()
            != self.NAVIGATION_REPEAT_MS
        ):
            self.navigation_timer.setInterval(
                self.NAVIGATION_REPEAT_MS
            )

    def _stop_continuous_navigation(
        self,
        direction: Optional[int] = None,
    ) -> None:
        timer = getattr(self, "navigation_timer", None)
        if timer is None:
            return
        if (
            direction is not None
            and direction != getattr(self, "navigation_direction", 0)
        ):
            return
        timer.stop()
        self.navigation_direction = 0

    def seek_to_index(self, index: int) -> None:
        if self.mode != "edit" or self.busy or not self.images:
            return
        target = max(0, min(int(index), len(self.images) - 1))
        if self._editor_is_dirty():
            self.progress.setValue(self.current_index)
            self.statusBar().showMessage(
                "Nhan Enter de luu hoac F de bo thay doi truoc khi keo thanh",
                5000,
            )
            return
        if target == self.current_index:
            return
        self.current_index = target
        self.show_current()

    def _preview_seek_position(self, index: int) -> None:
        if self.mode == "edit" and self.images:
            self.position_label.setText(
                "{} / {}".format(
                    max(0, min(index, len(self.images) - 1)) + 1,
                    len(self.images),
                )
            )

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.busy = busy
        if busy:
            self._stop_continuous_navigation()
        self.commit_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.remove_box_button.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message)

    def _update_schema_status(self) -> None:
        if self.mode == "edit":
            if self.editor_index is None:
                self.schema_badge.setText("Chua mo dataset YOLO")
                self.schema_badge.setStyleSheet(
                    "background:#1E293B;color:#CBD5E1;"
                )
            else:
                self.schema_badge.setText(
                    "{} lop - sua YOLO".format(
                        len(self.editor_index.class_names)
                    )
                )
                self.schema_badge.setStyleSheet(
                    "background:#172554;color:#93C5FD;"
                )
            self.migrate_button.setVisible(False)
            return
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
        if self.export_task is not None:
            if self.export_progress_dialog is not None:
                self.export_progress_dialog.request_cancel()
            event.ignore()
            return
        self._stop_continuous_navigation()
        self.settings.sync()
        super().closeEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_D):
            if not event.isAutoRepeat():
                self._start_continuous_navigation(
                    -1 if key == Qt.Key_A else 1
                )
            event.accept()
            return
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            if self.mode != "edit":
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
        if key == Qt.Key_Backspace and self.mode == "edit":
            self.remove_active_box()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        key = event.key()
        if key in (Qt.Key_A, Qt.Key_D):
            if not event.isAutoRepeat():
                self._stop_continuous_navigation(
                    -1 if key == Qt.Key_A else 1
                )
            event.accept()
            return
        super().keyReleaseEvent(event)

    def event(self, event) -> bool:  # type: ignore[no-untyped-def]
        if event.type() == QEvent.WindowDeactivate:
            self._stop_continuous_navigation()
        return super().event(event)
