from collections import OrderedDict
from pathlib import Path
import random
from typing import Dict, List, Optional

from PIL import Image, ImageOps
from PySide6.QtCore import QEvent, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QImageReader, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
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

from .busy_overlay import BusyOverlay
from .canvas import AnnotationCanvas
from .classification_editor import (
    ClassificationDatasetEditor,
    scan_classification_dataset,
)
from .constants import (
    APP_NAME,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_NAMES,
    DEFAULT_CLASSIFICATION_ROOT,
    DEFAULT_DETECTION_ROOT,
    SPLITS,
)
from .dataset import DatasetError, DatasetManager
from .export_progress_dialog import ExportProgressDialog
from .export_settings_dialog import (
    ExportPreviewItem,
    ExportSettingsDialog,
)
from .migration import apply_migration
from .models import (
    BBox,
    ClassificationDatasetIndex,
    ClassificationSample,
    ClassificationUndoResult,
    DatasetPaths,
    MigrationReport,
    YoloUndoResult,
)
from .scanner import scan_images
from .schema import audit_schema, initialize_empty_datasets
from .seek_slider import SeekSlider
from .settings_dialog import SettingsDialog
from .simple_classification_editor import (
    SimpleClassificationEditor,
    SimpleClassificationExportReport,
    StagedSimpleClassificationEditor,
    scan_simple_classification_dataset,
)
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
    CLASSIFICATION_MODES = ("class_edit", "simple_class")

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
        self.export_crop_padding = float(
            self.settings.value(
                "datasets/export_crop_padding",
                self.crop_padding,
            )
        )
        output_root_value = str(
            self.settings.value("datasets/output_root", "")
        ).strip()
        self.output_root = Path(output_root_value) if output_root_value else None
        self.source_root = Path(
            self.settings.value("source/root", "")
        ) if self.settings.value("source/root", "") else None
        simple_root_value = str(
            self.settings.value("simple_classification/root", "")
        ).strip()
        self.simple_classification_root = (
            Path(simple_root_value) if simple_root_value else None
        )
        simple_export_parent_value = str(
            self.settings.value("simple_classification/export_parent", "")
        ).strip()
        self.simple_export_parent = (
            Path(simple_export_parent_value)
            if simple_export_parent_value
            else None
        )
        self.simple_deferred_destination: Optional[Path] = None
        self._reverting_mode_change = False

        self.manager = self._make_manager()
        self.mode = "label"
        self.active_class_names = tuple(CLASS_NAMES)
        self.editor_index = None
        self.editor_manager = None
        self.editor_samples_by_path = {}
        self.editor_all_images: List[Path] = []
        self.editor_split_filter = "all"
        self.editor_class_filter = "all"
        self.editor_image_classes: Dict[Path, frozenset[int]] = {}
        self.editor_original_annotations = ()
        self.classification_index = None
        self.classification_manager = None
        self.classification_samples_by_path: Dict[
            Path,
            ClassificationSample,
        ] = {}
        self.classification_all_images: List[Path] = []
        self.classification_original_class_id = -1
        self.images: List[Path] = []
        self.current_index = 0
        self.current_path: Optional[Path] = None
        self.selected_class = -1
        self.busy = False
        self.closing = False
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
        self.export_destination_dialog: Optional[QFileDialog] = None

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
        subtitle = QLabel("Gan nhan va kiem tra dataset anh")
        subtitle.setObjectName("muted")
        side.addWidget(title)
        side.addWidget(subtitle)

        mode_label = QLabel("CHE DO")
        mode_label.setObjectName("sectionLabel")
        side.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Gan nhan anh moi", "label")
        self.mode_combo.addItem("Xem / sua dataset YOLO cu", "edit")
        self.mode_combo.addItem("Kiem tra / sua class_f", "class_edit")
        self.mode_combo.addItem(
            "Data phan loai don gian (thu muc class)",
            "simple_class",
        )
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        side.addWidget(self.mode_combo)

        self.source_label = QLabel("THU MUC NGUON")
        self.source_label.setObjectName("sectionLabel")
        side.addWidget(self.source_label)
        source_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Chon thu muc anh can gan lai nhan")
        self.source_edit.returnPressed.connect(self.scan_source)
        self.source_browse = QToolButton()
        self.source_browse.setText("...")
        self.source_browse.setToolTip("Chon thu muc (Ctrl+O)")
        self.source_browse.clicked.connect(self.choose_source)
        source_row.addWidget(self.source_edit)
        source_row.addWidget(self.source_browse)
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

        self.editor_split_status = QLabel("Split hien tai: -")
        self.editor_split_status.setObjectName("muted")
        self.editor_split_status.setWordWrap(True)
        self.editor_split_status.setToolTip(
            "Split goc cua anh dang xem trong dataset YOLO."
        )
        self.editor_split_status.setVisible(False)
        side.addWidget(self.editor_split_status)

        self.editor_split_filter_label = QLabel("XEM THEO SPLIT")
        self.editor_split_filter_label.setObjectName("sectionLabel")
        self.editor_split_filter_label.setVisible(False)
        side.addWidget(self.editor_split_filter_label)
        self.editor_split_filter_combo = QComboBox()
        self.editor_split_filter_combo.addItem(
            "Tat ca (train + val + test)",
            "all",
        )
        self.editor_split_filter_combo.addItem("Chi train", "train")
        self.editor_split_filter_combo.addItem("Chi val", "val")
        self.editor_split_filter_combo.addItem("Chi test", "test")
        self.editor_split_filter_combo.setToolTip(
            "Chi loc danh sach dang xem de dieu tra leak; export van dung "
            "toan bo dataset YOLO da chinh."
        )
        self.editor_split_filter_combo.currentIndexChanged.connect(
            self._editor_split_filter_changed
        )
        self.editor_split_filter_combo.setVisible(False)
        side.addWidget(self.editor_split_filter_combo)

        self.editor_class_filter_label = QLabel("XEM THEO CLASS")
        self.editor_class_filter_label.setObjectName("sectionLabel")
        self.editor_class_filter_label.setVisible(False)
        side.addWidget(self.editor_class_filter_label)
        self.editor_class_filter_combo = QComboBox()
        self.editor_class_filter_combo.addItem("Tat ca class", "all")
        self.editor_class_filter_combo.setToolTip(
            "Chi loc danh sach dang xem theo class co trong label YOLO da "
            "luu; cac nut class ben duoi van dung de sua nhan."
        )
        self.editor_class_filter_combo.currentIndexChanged.connect(
            self._editor_class_filter_changed
        )
        self.editor_class_filter_combo.setVisible(False)
        side.addWidget(self.editor_class_filter_combo)

        self.simple_save_mode_label = QLabel("CHE DO LUU")
        self.simple_save_mode_label.setObjectName("sectionLabel")
        self.simple_save_mode_label.setVisible(False)
        side.addWidget(self.simple_save_mode_label)
        self.simple_deferred_toggle = QPushButton(
            "Sua truc tiep dataset goc"
        )
        self.simple_deferred_toggle.setCheckable(True)
        self.simple_deferred_toggle.setProperty("saveModeToggle", True)
        self.simple_deferred_toggle.setToolTip(
            "Bat de Enter/DEL chi xep thay doi trong bo nho; dataset nguon "
            "khong bi sua."
        )
        self.simple_deferred_toggle.toggled.connect(
            self._simple_deferred_toggled
        )
        self.simple_deferred_toggle.setVisible(False)
        side.addWidget(self.simple_deferred_toggle)

        self.simple_destination_label = QLabel("DATASET MOI SE TAO")
        self.simple_destination_label.setObjectName("sectionLabel")
        self.simple_destination_label.setVisible(False)
        simple_destination_row = QHBoxLayout()
        self.simple_destination_edit = QLineEdit()
        self.simple_destination_edit.setPlaceholderText(
            "Duong dan dataset moi (phai rong/chua ton tai)"
        )
        self.simple_destination_edit.returnPressed.connect(
            self.apply_simple_deferred_destination
        )
        self.simple_destination_edit.textChanged.connect(
            lambda _text: self._sync_simple_deferred_ui()
        )
        self.simple_destination_edit.setVisible(False)
        self.simple_destination_browse = QToolButton()
        self.simple_destination_browse.setText("...")
        self.simple_destination_browse.setToolTip(
            "Chon thu muc cha; tool tu tao ten dataset moi"
        )
        self.simple_destination_browse.clicked.connect(
            self.choose_simple_deferred_destination
        )
        self.simple_destination_browse.setVisible(False)
        simple_destination_row.addWidget(self.simple_destination_edit)
        simple_destination_row.addWidget(self.simple_destination_browse)
        self.simple_deferred_hint = QLabel("")
        self.simple_deferred_hint.setObjectName("muted")
        self.simple_deferred_hint.setWordWrap(True)
        self.simple_deferred_hint.setVisible(False)
        self.simple_publish_button = QPushButton(
            "XAC NHAN  Tao dataset da sua"
        )
        self.simple_publish_button.setObjectName("primaryButton")
        self.simple_publish_button.clicked.connect(
            self.export_simple_deferred_dataset
        )
        self.simple_publish_button.setVisible(False)

        self.export_button = QPushButton(
            "Xuat yolo_f + class_f"
        )
        self.export_button.clicked.connect(self.export_editor_dataset)
        self.export_button.setVisible(False)
        side.addWidget(self.export_button)

        self.class_label = QLabel("CHON CLASS (PHIM 1-5)")
        self.class_label.setObjectName("sectionLabel")
        side.addWidget(self.class_label)
        self.class_button_container = QWidget()
        self.class_button_layout = QVBoxLayout(self.class_button_container)
        self.class_button_layout.setContentsMargins(0, 0, 0, 0)
        self.class_button_layout.setSpacing(12)
        side.addWidget(self.class_button_container)
        self._ensure_class_button_count(len(CLASS_NAMES))
        side.addWidget(self.simple_destination_label)
        side.addLayout(simple_destination_row)
        side.addWidget(self.simple_deferred_hint)
        side.addWidget(self.simple_publish_button)

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
        self.bbox_label.setWordWrap(True)
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
        self._configure_class_buttons(CLASS_NAMES)
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
        self.busy_overlay = BusyOverlay(root)
        self.busy_overlay_message = ""
        self.busy_overlay_timer = QTimer(self)
        self.busy_overlay_timer.setSingleShot(True)
        self.busy_overlay_timer.timeout.connect(
            self._show_delayed_busy_overlay
        )

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
            QPushButton[saveModeToggle="true"]:checked {
                background: #123524;
                border: 2px solid #22C55E;
                color: #BBF7D0;
                font-weight: 700;
            }
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
        if self.mode == "edit":
            title = "Chon dataset YOLO co data.yaml"
        elif self.mode == "class_edit":
            title = "Chon thu muc class_f"
        elif self.mode == "simple_class":
            title = "Chon thu muc data co cac thu muc class con"
        else:
            title = "Chon thu muc anh nguon"
        folder = QFileDialog.getExistingDirectory(
            self,
            title,
            self.source_edit.text() or str(Path.home()),
        )
        if folder:
            self.source_edit.setText(folder)
            self.scan_source()

    def _mode_changed(self) -> None:
        if self._reverting_mode_change:
            return
        mode = str(self.mode_combo.currentData())
        if mode == self.mode:
            return
        if (
            self.mode == "simple_class"
            and self._simple_session_has_unsaved_changes()
            and not self._confirm_discard_simple_staging(
                "doi sang che do khac"
            )
        ):
            self._reverting_mode_change = True
            try:
                self.mode_combo.setCurrentIndex(
                    self.mode_combo.findData(self.mode)
                )
            finally:
                self._reverting_mode_change = False
            return
        self.mode = mode
        self._stop_continuous_navigation()
        self.images = []
        self.current_index = 0
        self.current_path = None
        self.editor_index = None
        self.editor_manager = None
        self.editor_samples_by_path = {}
        self.editor_all_images = []
        self.editor_image_classes = {}
        self._set_editor_split_filter_ui("all")
        self._set_editor_class_filter_ui("all")
        self.editor_original_annotations = ()
        self.classification_index = None
        self.classification_manager = None
        self.classification_samples_by_path = {}
        self.classification_all_images = []
        self.classification_original_class_id = -1
        self.image_cache.clear()
        self.canvas.clear_image()
        self.canvas.set_annotation_enabled(
            mode not in self.CLASSIFICATION_MODES
        )
        self.file_label.setText("Chua co anh")
        self.path_label.clear()
        self.queue_label.setText("0 anh")
        self.position_label.setText("0 / 0")
        self.progress.setValue(0)

        editing = mode == "edit"
        class_f_editing = mode == "class_edit"
        simple_editing = mode == "simple_class"
        classification_editing = mode in self.CLASSIFICATION_MODES
        filterable = editing or classification_editing
        split_filterable = editing or class_f_editing

        if editing:
            self.source_label.setText("DATASET YOLO CU")
            self.source_edit.setPlaceholderText("Chon thu muc chua data.yaml")
            self.scan_button.setText("Mo dataset YOLO")
            self.source_edit.clear()
        elif class_f_editing:
            self.source_label.setText("DATASET CLASS_F")
            self.source_edit.setPlaceholderText(
                "Chon thu muc class_f co train/val/test"
            )
            self.scan_button.setText("Mo class_f")
            self.source_edit.clear()
        elif simple_editing:
            self.source_label.setText("DATA PHAN LOAI DON GIAN")
            self.source_edit.setPlaceholderText(
                "Chon thu muc data; moi thu muc con la mot class"
            )
            self.scan_button.setText("Mo data don gian")
            self.source_edit.setText(
                str(self.simple_classification_root)
                if self.simple_classification_root is not None
                else ""
            )
        else:
            self.source_label.setText("THU MUC NGUON")
            self.source_edit.setPlaceholderText(
                "Chon thu muc anh can gan lai nhan"
            )
            self.scan_button.setText("Quet anh")
            self.source_edit.clear()

        self.destination_label.setVisible(not filterable)
        self.destination_edit.setVisible(not filterable)
        self.destination_browse.setVisible(not filterable)
        self.destination_hint.setVisible(not filterable)
        self.split_label.setVisible(not filterable)
        self.split_combo.setVisible(not filterable)
        self.editor_split_status.setVisible(filterable)
        self.editor_split_filter_label.setVisible(split_filterable)
        self.editor_split_filter_combo.setVisible(split_filterable)
        self.editor_class_filter_label.setVisible(filterable)
        self.editor_class_filter_combo.setVisible(filterable)
        self.undo_button.setVisible(True)
        self.remove_box_button.setVisible(editing)
        self.export_button.setVisible(editing)
        self.next_box_action.setEnabled(editing)
        self.previous_box_action.setEnabled(editing)
        self.progress.setEnabled(filterable)
        if editing:
            self.commit_button.setText("ENTER  Ap dung thay doi")
            self.reset_button.setText("F  Ve nhan goc")
            self.delete_button.setText("DEL  Xoa anh + nhan")
            self.help_text.setText(
                "Click box: chon object   |   Keo/resize: sua box   |   "
                "Tab/Shift+Tab: doi box   |   Keo vung trong: them box   |   "
                "Backspace: xoa box   |   Ctrl+Z: hoan tac   |   "
                "Giu A/D: anh truoc/sau"
            )
        elif classification_editing:
            self.commit_button.setText("ENTER  Ap dung class")
            self.reset_button.setText("F  Ve class goc")
            self.delete_button.setText(
                "DEL  Loai bo khoi data"
                if simple_editing
                else "DEL  Xoa anh khoi class_f"
            )
            self.help_text.setText(
                "Phim 1-9/click: doi class   |   ENTER: ap dung   |   "
                "F: ve class goc   |   DEL: loai anh   |   "
                "Ctrl+Z: hoan tac   |   Wheel: zoom   |   "
                "Space+drag: pan   |   Giu A/D: anh truoc/sau"
            )
        else:
            self.commit_button.setText("ENTER  Luu va chuyen anh")
            self.reset_button.setText("F  Lam lai")
            self.delete_button.setText("DEL  Loai bo")
            self.help_text.setText(
                "Keo chuot: tao bbox   |   Keo trong khung: di chuyen   |   "
                "Keo diem vuong: resize   |   Wheel: zoom   |   "
                "Space+drag: pan   |   Giu A/D: anh truoc/sau"
            )
        self.class_label.setText(
            "CHON CLASS (PHIM 1-9 HOAC CLICK)"
            if simple_editing
            else "CHON CLASS (PHIM 1-5)"
        )
        self._configure_class_buttons(() if simple_editing else CLASS_NAMES)
        self._update_schema_status()
        self._sync_simple_deferred_ui()

    def _staged_simple_manager(
        self,
    ) -> Optional[StagedSimpleClassificationEditor]:
        manager = self.classification_manager
        return (
            manager
            if isinstance(manager, StagedSimpleClassificationEditor)
            else None
        )

    def _simple_session_has_unsaved_changes(self) -> bool:
        manager = self._staged_simple_manager()
        return bool(
            self.mode == "simple_class"
            and (
                (manager is not None and manager.has_unexported_changes)
                or self._classification_is_dirty()
            )
        )

    def _confirm_discard_simple_staging(self, action: str) -> bool:
        answer = QMessageBox.question(
            self,
            "Bo thay doi chua xuat?",
            (
                "Phien luu ban sao con thay doi chua duoc tao thanh dataset "
                "moi.\nDataset nguon van nguyen ven.\n\n"
                "Bo cac thay doi staging de {}?"
            ).format(action),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _simple_deferred_toggled(self, checked: bool) -> None:
        if self.busy:
            return
        if self.mode != "simple_class":
            self._sync_simple_deferred_ui()
            return
        if (
            not checked
            and self._simple_session_has_unsaved_changes()
            and not self._confirm_discard_simple_staging(
                "tat che do luu ban sao"
            )
        ):
            self.simple_deferred_toggle.blockSignals(True)
            self.simple_deferred_toggle.setChecked(True)
            self.simple_deferred_toggle.blockSignals(False)
            self._sync_simple_deferred_ui()
            return
        if self.classification_index is not None:
            try:
                if checked:
                    index = self._current_simple_classification_index()
                    manager = StagedSimpleClassificationEditor(index)
                    self.classification_index = index
                else:
                    manager = SimpleClassificationEditor(
                        self.classification_index
                    )
                self._install_simple_classification_manager(manager)
            except Exception as error:
                self._show_error(str(error))
                self.simple_deferred_toggle.blockSignals(True)
                self.simple_deferred_toggle.setChecked(not checked)
                self.simple_deferred_toggle.blockSignals(False)
                self._sync_simple_deferred_ui()
                return
        if not checked:
            self.simple_deferred_destination = None
            self.simple_destination_edit.clear()
        self._sync_simple_deferred_ui()
        self._update_schema_status()
        if checked and not self.simple_destination_edit.text().strip():
            QTimer.singleShot(0, self.choose_simple_deferred_destination)

    def _current_simple_classification_index(
        self,
    ) -> ClassificationDatasetIndex:
        if self.classification_index is None:
            raise RuntimeError("Chua mo data phan loai don gian")
        samples = tuple(
            sorted(
                self.classification_samples_by_path.values(),
                key=lambda sample: (
                    sample.class_id,
                    str(sample.image_path).casefold(),
                ),
            )
        )
        return ClassificationDatasetIndex(
            root=self.classification_index.root,
            data_yaml=None,
            class_names=tuple(self.classification_index.class_names),
            samples=samples,
        )

    def _install_simple_classification_manager(self, manager) -> None:
        if self.classification_index is None:
            return
        preferred_path = self.current_path
        self.classification_manager = manager
        self.classification_samples_by_path = {
            sample.image_path: sample
            for sample in self.classification_index.samples
        }
        self.classification_all_images = [
            sample.image_path for sample in self.classification_index.samples
        ]
        self._refresh_simple_class_counts()
        self._apply_classification_filters(
            preferred_path,
            fallback_index=self.current_index,
        )

    def _sync_simple_deferred_ui(self) -> None:
        simple_mode = self.mode == "simple_class"
        enabled = simple_mode and self.simple_deferred_toggle.isChecked()
        self.simple_save_mode_label.setVisible(simple_mode)
        self.simple_deferred_toggle.setVisible(simple_mode)
        self.simple_destination_label.setVisible(enabled)
        self.simple_destination_edit.setVisible(enabled)
        self.simple_destination_browse.setVisible(enabled)
        self.simple_deferred_hint.setVisible(enabled)
        self.simple_publish_button.setVisible(enabled)
        self.simple_deferred_toggle.setText(
            "DANG BAT  Luu thanh dataset moi"
            if enabled
            else "Sua truc tiep dataset goc"
        )
        manager = self._staged_simple_manager()
        if enabled and manager is None:
            hint = "Mo dataset de bat dau phien sua khong cham vao data nguon."
        elif enabled:
            hint = (
                "Nguon giu nguyen | {} doi class | {} loai bo. "
                "Enter/DEL chi xep thay doi trong bo nho."
            ).format(manager.changed_count, manager.deleted_count)
        else:
            hint = ""
        self.simple_deferred_hint.setText(hint)
        pending = bool(manager is not None and manager.has_unexported_changes)
        self.simple_publish_button.setText(
            "XAC NHAN  Tao dataset moi ({} thay doi)".format(
                manager.staged_change_count if manager is not None else 0
            )
        )
        self.simple_publish_button.setEnabled(
            enabled
            and not self.busy
            and pending
            and bool(self.simple_destination_edit.text().strip())
            and not self._classification_is_dirty()
        )
        self.simple_destination_edit.setEnabled(enabled and not self.busy)
        self.simple_destination_browse.setEnabled(enabled and not self.busy)
        self.simple_deferred_toggle.setEnabled(not self.busy)
        if simple_mode:
            self.commit_button.setText(
                "ENTER  Xep thay doi va sang anh"
                if enabled
                else "ENTER  Ap dung class"
            )
            self.delete_button.setText(
                "DEL  Loai khoi ban dataset moi"
                if enabled
                else "DEL  Loai bo khoi data"
            )
            self.help_text.setText(
                (
                    "Phim 1-9/click: doi class   |   ENTER/DEL: xep thay "
                    "doi   |   XAC NHAN: tao dataset moi   |   F: ve class "
                    "hien tai   |   Ctrl+Z: hoan tac   |   Giu A/D: dieu huong"
                )
                if enabled
                else (
                    "Phim 1-9/click: doi class   |   ENTER: ap dung   |   "
                    "F: ve class goc   |   DEL: loai anh   |   "
                    "Ctrl+Z: hoan tac   |   Wheel: zoom   |   "
                    "Space+drag: pan   |   Giu A/D: anh truoc/sau"
                )
            )

    def choose_simple_deferred_destination(self) -> None:
        if (
            self.mode != "simple_class"
            or not self.simple_deferred_toggle.isChecked()
        ):
            return
        if self.simple_export_parent is not None:
            initial = self.simple_export_parent
        elif self.classification_index is not None:
            initial = self.classification_index.root.parent
        else:
            initial = Path.home()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chon thu muc cha de tao dataset moi",
            str(initial),
        )
        if not folder:
            self._sync_simple_deferred_ui()
            return
        try:
            parent = Path(folder).expanduser().resolve()
            destination = self._suggest_simple_deferred_destination(parent)
        except Exception as error:
            self.simple_deferred_destination = None
            self._sync_simple_deferred_ui()
            self._show_error(str(error))
            return
        self.simple_export_parent = parent
        self.settings.setValue(
            "simple_classification/export_parent",
            str(parent),
        )
        self.simple_deferred_destination = destination
        self.simple_destination_edit.setText(str(destination))
        self.simple_destination_edit.setCursorPosition(0)
        self.simple_destination_edit.setToolTip(str(destination))
        self._sync_simple_deferred_ui()
        self.statusBar().showMessage(
            "Dataset moi se duoc tao tai {}".format(destination),
            5000,
        )

    def _suggest_simple_deferred_destination(self, parent: Path) -> Path:
        source_name = (
            self.classification_index.root.name
            if self.classification_index is not None
            else "data"
        )
        base_name = "{}_edited".format(source_name)
        candidate = parent / base_name
        for index in range(10000):
            value = (
                candidate
                if index == 0
                else parent / "{}_{:03d}".format(base_name, index)
            )
            if not value.exists():
                return value.resolve(strict=False)
        raise RuntimeError("Khong tao duoc ten dataset dich khong trung")

    def apply_simple_deferred_destination(self) -> None:
        text = self.simple_destination_edit.text().strip()
        if not text:
            self.simple_deferred_destination = None
            self._sync_simple_deferred_ui()
            self.statusBar().showMessage("Chon duong dan dataset moi", 3000)
            return
        try:
            destination = Path(text).expanduser().resolve(strict=False)
            invalid_existing = destination.exists() and (
                not destination.is_dir() or any(destination.iterdir())
            )
        except (OSError, RuntimeError) as error:
            self.simple_deferred_destination = None
            self._sync_simple_deferred_ui()
            self._show_error(
                "Khong doc duoc thu muc dataset dich: {}".format(error)
            )
            return
        if invalid_existing:
            self._show_error(
                "Thu muc dataset dich phai rong hoac chua ton tai"
            )
            self.simple_deferred_destination = None
            self._sync_simple_deferred_ui()
            return
        self.simple_deferred_destination = destination
        self.simple_destination_edit.setText(str(destination))
        self.simple_destination_edit.setToolTip(str(destination))
        self._sync_simple_deferred_ui()
        self.statusBar().showMessage(
            "Da chon dataset moi: {}".format(destination),
            4000,
        )

    def _class_color(self, class_id: int) -> str:
        if 0 <= class_id < len(CLASS_COLORS):
            return CLASS_COLORS[class_id]
        hue = (class_id * 137 + 17) % 360
        return QColor.fromHsv(hue, 185, 235).name()

    def _ensure_class_button_count(self, count: int) -> None:
        while len(self.class_buttons) < count:
            index = len(self.class_buttons)
            button = QPushButton()
            button.setCheckable(True)
            button.setProperty("classButton", True)
            button.setMinimumHeight(58)
            button.clicked.connect(
                lambda checked=False, value=index: self.select_class(value)
            )
            self.class_buttons.append(button)
            self.class_button_layout.addWidget(button)

    def _simple_class_counts(self):
        if self.mode != "simple_class" or self.classification_manager is None:
            return ()
        counts = getattr(self.classification_manager, "class_counts", ())
        return tuple(int(value) for value in counts)

    def _update_class_button_texts(self) -> None:
        counts = self._simple_class_counts()
        canonical = self.active_class_names == CLASS_NAMES
        for index, button in enumerate(self.class_buttons):
            visible = index < len(self.active_class_names)
            button.setVisible(visible)
            if not visible:
                button.setChecked(False)
                continue
            name = self.active_class_names[index]
            color = self._class_color(index)
            button.setProperty("classColor", color)
            if self.mode == "simple_class":
                count = counts[index] if index < len(counts) else 0
                button.setText(
                    "{}   {}\n     {} anh".format(index + 1, name, count)
                )
            elif canonical:
                button.setText(
                    "{}   {}\n     {}".format(
                        index + 1,
                        CLASS_LABELS[index],
                        name,
                    )
                )
            else:
                button.setText("{}   {}".format(index + 1, name))
            button.setToolTip(
                "Phim {}\nClass: {}".format(index + 1, name)
                if index < 9
                else "Click de chon class\nClass: {}".format(name)
            )

    def _configure_class_buttons(self, names) -> None:
        class_names = tuple(str(name) for name in names)
        if self.mode != "simple_class" and len(class_names) > len(CLASS_NAMES):
            raise YoloEditorError(
                "UI sua YOLO/class_f ho tro toi da {} class".format(
                    len(CLASS_NAMES)
                )
            )
        self._ensure_class_button_count(len(class_names))
        self.active_class_names = class_names
        for button in self.class_buttons:
            button.setChecked(False)
        self._update_class_button_texts()
        self.canvas.set_class_catalog(
            self.active_class_names,
            tuple(
                self._class_color(index)
                for index in range(len(self.active_class_names))
            ),
        )
        self.selected_class = -1

    def _refresh_simple_class_counts(self) -> None:
        if self.mode != "simple_class":
            return
        self._update_class_button_texts()
        self._configure_editor_class_filter(self.active_class_names)

    def _configure_editor_class_filter(self, names) -> None:
        current = self.editor_class_filter
        counts = self._simple_class_counts()
        combo = self.editor_class_filter_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(
            (
                "Tat ca class ({} anh)".format(sum(counts))
                if counts
                else "Tat ca class"
            ),
            "all",
        )
        for index, name in enumerate(names):
            label = (
                CLASS_LABELS[index]
                if tuple(names) == CLASS_NAMES
                else str(name)
            )
            combo.addItem(
                (
                    "{}  {} ({} anh)".format(
                        index + 1,
                        label,
                        counts[index],
                    )
                    if index < len(counts)
                    else "{}  {}".format(index + 1, label)
                ),
                index,
            )
        selected_index = combo.findData(current)
        if selected_index < 0:
            selected_index = combo.findData("all")
            current = "all"
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)
        self.editor_class_filter = current

    def _read_sample_class_ids(self, sample) -> frozenset[int]:
        if self.editor_index is None:
            return frozenset()
        try:
            lines = sample.label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        except OSError:
            return frozenset()
        class_ids = set()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            try:
                class_id = int(parts[0])
            except (IndexError, ValueError):
                continue
            if 0 <= class_id < len(self.editor_index.class_names):
                class_ids.add(class_id)
        return frozenset(class_ids)

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
        if self.busy:
            return
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
        if self.mode == "class_edit":
            self._scan_classification_dataset(root)
            return
        if self.mode == "simple_class":
            if (
                self._simple_session_has_unsaved_changes()
                and not self._confirm_discard_simple_staging(
                    "mo dataset khac"
                )
            ):
                if self.classification_index is not None:
                    self.source_edit.setText(
                        str(self.classification_index.root)
                    )
                return
            self._scan_simple_classification_dataset(root)
            return
        self.source_root = root.resolve()
        self.settings.setValue("source/root", str(self.source_root))
        self.scan_button.setText("Dang quet...")
        self._set_busy(
            True,
            "Dang quet thu muc va cac thu muc con...",
            delayed_overlay=True,
        )
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
                self._set_busy(False),
                self.scan_button.setText("Quet anh"),
            )
        )
        self._start_task(task)

    def _scan_simple_classification_dataset(self, root: Path) -> None:
        self.scan_button.setText("Dang mo data...")
        self._set_busy(
            True,
            "Dang doc cac thu muc class va dem anh...",
            delayed_overlay=True,
        )
        task = FunctionTask(scan_simple_classification_dataset, root)
        task.signals.succeeded.connect(
            self._simple_classification_scan_finished
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda: (
                self._set_busy(False),
                self.scan_button.setText("Mo data don gian"),
            )
        )
        self._start_task(task)

    def _simple_classification_scan_finished(self, result: object) -> None:
        try:
            manager = (
                StagedSimpleClassificationEditor(result)
                if self.simple_deferred_toggle.isChecked()
                else SimpleClassificationEditor(result)
            )
        except Exception as error:
            self._show_error(str(error))
            return
        self.classification_index = result
        self.classification_manager = manager
        self.classification_samples_by_path = {
            sample.image_path: sample for sample in result.samples
        }
        self.classification_all_images = [
            sample.image_path for sample in result.samples
        ]
        self._set_editor_split_filter_ui("all")
        self._set_editor_class_filter_ui("all")
        self._configure_class_buttons(result.class_names)
        self._configure_editor_class_filter(result.class_names)
        self.images = list(self.classification_all_images)
        self.current_index = 0
        self.image_cache.clear()
        self.simple_classification_root = result.root
        self.settings.setValue(
            "simple_classification/root",
            str(result.root),
        )
        self.source_edit.setCursorPosition(0)
        self.source_edit.setToolTip(str(result.root))
        self.queue_label.setText(self._classification_queue_text())
        self.progress.setRange(0, max(0, len(self.images) - 1))
        self.progress.setValue(0)
        self.schema_badge.setText(
            "{} class - data don gian".format(len(result.class_names))
        )
        self.schema_badge.setStyleSheet(
            "background:#164E63;color:#A5F3FC;"
        )
        self.migrate_button.setVisible(False)
        self.statusBar().showMessage(
            "Da mo {} anh trong {} class tu {}".format(
                len(self.images),
                len(result.class_names),
                result.root,
            ),
            5000,
        )
        self._sync_simple_deferred_ui()
        self._update_schema_status()
        self.show_current()

    def _scan_classification_dataset(self, root: Path) -> None:
        self.scan_button.setText("Dang mo class_f...")
        self._set_busy(
            True,
            "Dang doc class_f va ghep split/class...",
            delayed_overlay=True,
        )
        task = FunctionTask(scan_classification_dataset, root)
        task.signals.succeeded.connect(self._classification_scan_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda: (
                self._set_busy(False),
                self.scan_button.setText("Mo class_f"),
            )
        )
        self._start_task(task)

    def _classification_scan_finished(self, result: object) -> None:
        if len(result.class_names) > len(CLASS_NAMES):
            self._show_error(
                "Dataset co {} class, UI hien tai ho tro toi da {}".format(
                    len(result.class_names),
                    len(CLASS_NAMES),
                )
            )
            return
        self.classification_index = result
        self.classification_manager = ClassificationDatasetEditor(result)
        self.classification_samples_by_path = {
            sample.image_path: sample for sample in result.samples
        }
        self._configure_class_buttons(result.class_names)
        self._configure_editor_class_filter(result.class_names)
        self.classification_all_images = [
            sample.image_path for sample in result.samples
        ]
        self._set_editor_split_filter_ui("all")
        self._set_editor_class_filter_ui("all")
        self.images = list(self.classification_all_images)
        self.current_index = 0
        self.image_cache.clear()
        self.queue_label.setText(self._classification_queue_text())
        self.progress.setRange(0, max(0, len(self.images) - 1))
        self.progress.setValue(0)
        self.schema_badge.setText(
            "{} lop - sua class_f".format(len(result.class_names))
        )
        self.schema_badge.setStyleSheet(
            "background:#312E81;color:#C4B5FD;"
        )
        self.migrate_button.setVisible(False)
        self.statusBar().showMessage(
            "Da mo {} anh class_f tu {}".format(
                len(self.images),
                result.root,
            ),
            5000,
        )
        self.show_current()

    def _scan_editor_dataset(self, root: Path) -> None:
        self.scan_button.setText("Dang mo dataset...")
        self._set_busy(
            True,
            "Dang doc data.yaml va ghep cap images/labels...",
            delayed_overlay=True,
        )
        task = FunctionTask(scan_yolo_dataset, root)
        task.signals.succeeded.connect(self._editor_scan_finished)
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(
            lambda: (
                self._set_busy(False),
                self.scan_button.setText("Mo dataset YOLO"),
            )
        )
        self._start_task(task)

    def _editor_scan_finished(self, result: object) -> None:
        if len(result.class_names) > len(CLASS_NAMES):
            self._show_error(
                "Dataset co {} class, UI hien tai ho tro toi da {}".format(
                    len(result.class_names),
                    len(CLASS_NAMES),
                )
            )
            return
        self.editor_index = result
        self.editor_manager = YoloDatasetEditor(result)
        self.editor_samples_by_path = {
            sample.image_path: sample for sample in result.samples
        }
        self._configure_class_buttons(result.class_names)
        self._configure_editor_class_filter(result.class_names)
        self.editor_all_images = [
            sample.image_path for sample in result.samples
        ]
        self.editor_image_classes = {
            sample.image_path: self._read_sample_class_ids(sample)
            for sample in result.samples
        }
        self._set_editor_split_filter_ui("all")
        self._set_editor_class_filter_ui("all")
        self.images = list(self.editor_all_images)
        self.current_index = 0
        self.image_cache.clear()
        self.queue_label.setText(self._editor_queue_text())
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
            missing = self.images[self.current_index]
            if self.mode == "edit":
                self._remove_editor_path_from_lists(missing)
            elif self.mode in self.CLASSIFICATION_MODES:
                self._remove_classification_path_from_lists(missing)
            else:
                self.images.pop(self.current_index)
            if self.current_index >= len(self.images):
                self.current_index = max(0, len(self.images) - 1)
        if not self.images:
            self.current_path = None
            self.canvas.clear_image()
            if self.mode in ("edit", *self.CLASSIFICATION_MODES):
                self.file_label.setText(
                    "Khong co anh trong bo loc {}".format(
                        self._editor_filter_title()
                    )
                )
            else:
                self.file_label.setText("Khong con anh")
            self.path_label.clear()
            self.position_label.setText("0 / 0")
            self.queue_label.setText(
                (
                    self._editor_queue_text()
                    if self.mode == "edit"
                    else self._classification_queue_text()
                )
                if self.mode in ("edit", *self.CLASSIFICATION_MODES)
                else "0 anh"
            )
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            if self.mode == "edit":
                self.editor_original_annotations = ()
                self._update_editor_split_status()
            elif self.mode in self.CLASSIFICATION_MODES:
                self.classification_original_class_id = -1
                self._update_classification_status()
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
                self._editor_queue_text()
                if self.mode == "edit"
                else self._classification_queue_text()
                if self.mode in self.CLASSIFICATION_MODES
                else "{} anh con lai".format(len(self.images))
            )
        )
        self.progress.setRange(0, max(0, len(self.images) - 1))
        self.progress.setValue(self.current_index)
        self.canvas.reset_annotation()
        if self.mode in self.CLASSIFICATION_MODES:
            self._load_classification_sample(self.current_path)
        self._load_image(self.current_path, display=True)
        if self.current_index + 1 < len(self.images):
            self._load_image(self.images[self.current_index + 1], display=False)
        if self.mode == "edit":
            self._update_editor_split_status()
        elif self.mode in self.CLASSIFICATION_MODES:
            self._update_classification_status()

    def _load_image(self, path: Path, display: bool) -> None:
        key = str(path)
        cached = self.image_cache.get(key)
        if cached is not None:
            self.image_cache.move_to_end(key)
            if display and self.current_path == path:
                self.canvas.set_image(cached)
                if self.mode == "edit":
                    self._load_editor_annotations(path, cached)
                elif self.mode in self.CLASSIFICATION_MODES:
                    self._load_classification_sample(path)
                self._bbox_changed(None)
            return
        if key in self.pending_loads:
            if display:
                self.canvas.set_loading(
                    "Dang tai {}...".format(path.name)
                )
            return
        self.pending_loads.add(key)
        if display:
            self.canvas.set_loading(
                "Dang tai {}...".format(path.name)
            )
            self.statusBar().showMessage("Dang tai {}...".format(path.name))
        task = FunctionTask(load_qimage, path)
        task.signals.succeeded.connect(
            lambda image, value=path: self._image_loaded(
                value,
                image,
            )
        )
        task.signals.failed.connect(
            lambda traceback_text, value=path: self._image_load_failed(
                value,
                traceback_text,
            )
        )
        task.signals.finished.connect(
            lambda value=key: self.pending_loads.discard(value)
        )
        self._start_task(task)

    def _image_loaded(self, path: Path, result: object) -> None:
        if self.closing:
            return
        image = result
        if not isinstance(image, QImage):
            return
        key = str(path)
        self.image_cache[key] = image
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > 8:
            self.image_cache.popitem(last=False)
        if self.current_path == path:
            self.canvas.set_image(image)
            if self.mode == "edit":
                self._load_editor_annotations(path, image)
            elif self.mode in self.CLASSIFICATION_MODES:
                self._load_classification_sample(path)
            self.statusBar().showMessage(
                "{} x {} px".format(image.width(), image.height()),
                3000,
            )

    def _image_load_failed(
        self,
        path: Path,
        traceback_text: str,
    ) -> None:
        if self.closing:
            return
        if self.current_path == path:
            self.canvas.clear_loading()
        self._background_failed(traceback_text)

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
        self.editor_image_classes[path] = frozenset(
            annotation.class_id for annotation in annotations
        )
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

    def _load_classification_sample(self, path: Path) -> None:
        sample = self.classification_samples_by_path.get(path)
        if sample is None:
            self.classification_original_class_id = -1
            self._update_classification_status()
            return
        self.classification_original_class_id = sample.class_id
        self._select_class_ui(sample.class_id)
        self._update_classification_status()

    def select_class(self, class_id: int) -> None:
        if self.busy:
            return
        if not 0 <= class_id < len(self.active_class_names):
            return
        self._select_class_ui(class_id)
        if self.mode == "edit":
            self.canvas.update_active_class(class_id)
            self._update_object_status()
        elif self.mode in self.CLASSIFICATION_MODES:
            self._update_classification_status()
            if self.mode == "simple_class":
                self._sync_simple_deferred_ui()

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
            self._class_color(class_id),
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

    def _update_classification_status(self) -> None:
        if self.mode not in self.CLASSIFICATION_MODES:
            return
        sample = self.classification_samples_by_path.get(self.current_path)
        if sample is None:
            self.editor_split_status.setText("Split hien tai: -")
            self.bbox_label.setText("Class: -")
            return
        original = self._class_filter_title(sample.class_id)
        selected = (
            self._class_filter_title(self.selected_class)
            if 0 <= self.selected_class < len(self.active_class_names)
            else original
        )
        if self.mode == "simple_class":
            counts = self._simple_class_counts()
            count = counts[sample.class_id] if sample.class_id < len(counts) else 0
            self.editor_split_status.setText(
                "Class hien tai: {}\n{} anh | Bo loc: {}".format(
                    sample.class_name,
                    count,
                    self._editor_filter_title(),
                )
            )
        else:
            self.editor_split_status.setText(
                "Split: {} | Class hien tai: {} | Bo loc: {}".format(
                    sample.split.upper(),
                    original,
                    self._editor_filter_title(),
                )
            )
        self.bbox_label.setText(
            "Class: {} -> {}{}".format(
                original,
                selected,
                " | CHUA LUU" if self._classification_is_dirty() else "",
            )
        )

    def _editor_is_dirty(self) -> bool:
        return (
            self.mode == "edit"
            and tuple(self.canvas.annotations)
            != tuple(self.editor_original_annotations)
        )

    def _classification_is_dirty(self) -> bool:
        return (
            self.mode in self.CLASSIFICATION_MODES
            and self.current_path is not None
            and 0 <= self.selected_class < len(self.active_class_names)
            and self.selected_class != self.classification_original_class_id
        )

    def _current_edit_is_dirty(self) -> bool:
        return self._editor_is_dirty() or self._classification_is_dirty()

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
        if self.mode in self.CLASSIFICATION_MODES:
            if 0 <= self.classification_original_class_id < len(
                self.active_class_names
            ):
                self._select_class_ui(self.classification_original_class_id)
            self._update_classification_status()
            if self.mode == "simple_class":
                self._sync_simple_deferred_ui()
            self.statusBar().showMessage(
                "Da khoi phuc class goc cua anh hien tai",
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
        if self.mode in self.CLASSIFICATION_MODES:
            self._update_classification_status()
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
        if self.mode in self.CLASSIFICATION_MODES:
            self._commit_classification_current()
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
        if tuple(annotations) == tuple(self.editor_original_annotations):
            if self.navigate(1):
                self.statusBar().showMessage(
                    "Nhan khong doi; da chuyen sang anh tiep theo",
                    3000,
                )
            else:
                self.statusBar().showMessage("Nhan khong doi", 3000)
            return
        self._set_busy(
            True,
            "Dang backup va ghi label YOLO...",
            delayed_overlay=True,
        )
        task = FunctionTask(
            self.editor_manager.save_annotations,
            sample,
            annotations,
            width,
            height,
        )
        task.signals.succeeded.connect(
            lambda _, value=sample.image_path, saved=annotations: (
                self._editor_save_finished(value, saved)
            )
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _editor_save_finished(self, path: Path, annotations) -> None:
        visible_before = list(self.images)
        try:
            saved_index = visible_before.index(path)
        except ValueError:
            saved_index = self.current_index
        next_path = (
            visible_before[saved_index + 1]
            if saved_index + 1 < len(visible_before)
            else path
        )
        self.editor_image_classes[path] = frozenset(
            annotation.class_id for annotation in annotations
        )
        self.editor_original_annotations = tuple(annotations)
        self.statusBar().showMessage(
            "Da ap dung {} object vao label".format(len(annotations)),
            4000,
        )
        if self.mode == "edit":
            self._apply_editor_filters(next_path, fallback_index=saved_index)

    def _commit_classification_current(self) -> None:
        if (
            self.classification_manager is None
            or self.current_path is None
            or self.selected_class < 0
        ):
            return
        sample = self.classification_samples_by_path.get(self.current_path)
        if sample is None:
            return
        if self.selected_class == sample.class_id:
            if self.navigate(1):
                self.statusBar().showMessage(
                    "Class khong doi; da chuyen sang anh tiep theo",
                    3000,
                )
            else:
                self.statusBar().showMessage("Class khong doi", 3000)
            return
        path = self.current_path
        class_id = self.selected_class
        staged_manager = self._staged_simple_manager()
        if staged_manager is not None:
            try:
                result = staged_manager.change_class(sample, class_id)
            except Exception as error:
                self._show_error(str(error))
                return
            self._classification_save_finished(path, result)
            return
        self._set_busy(
            True,
            (
                "Dang di chuyen anh sang thu muc class..."
                if self.mode == "simple_class"
                else "Dang doi class va cap nhat class_f..."
            ),
            delayed_overlay=True,
        )
        task = FunctionTask(
            self.classification_manager.change_class,
            sample,
            class_id,
        )
        task.signals.succeeded.connect(
            lambda result, value=path: self._classification_save_finished(
                value,
                result,
            )
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _classification_save_finished(
        self,
        old_path: Path,
        result: object,
    ) -> None:
        if not isinstance(result, ClassificationSample):
            return
        visible_before = list(self.images)
        try:
            saved_index = visible_before.index(old_path)
        except ValueError:
            saved_index = self.current_index
        next_path = (
            visible_before[saved_index + 1]
            if saved_index + 1 < len(visible_before)
            else result.image_path
        )
        self._replace_classification_sample(old_path, result)
        self.statusBar().showMessage(
            (
                "Da xep doi class sang {}; dataset nguon chua bi sua"
                if self._staged_simple_manager() is not None
                else "Da doi class sang {}"
            ).format(self._class_filter_title(result.class_id)),
            4000,
        )
        if self.mode in self.CLASSIFICATION_MODES:
            self._refresh_simple_class_counts()
            self._apply_classification_filters(
                next_path,
                fallback_index=saved_index,
            )
            if self.mode == "simple_class":
                self._sync_simple_deferred_ui()

    def delete_current(self) -> None:
        if self.busy or self.current_path is None:
            return
        if self.mode == "edit":
            self._delete_editor_current()
            return
        if self.mode in self.CLASSIFICATION_MODES:
            self._delete_classification_current()
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
        self._set_busy(
            True,
            "Dang archive anh va label...",
            delayed_overlay=True,
        )
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
        self._remove_editor_path_from_lists(path)
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

    def _delete_classification_current(self) -> None:
        if self.classification_manager is None or self.current_path is None:
            return
        sample = self.classification_samples_by_path.get(self.current_path)
        if sample is None:
            return
        path = self.current_path
        staged_manager = self._staged_simple_manager()
        if staged_manager is not None:
            try:
                staged_manager.delete_sample(sample)
            except Exception as error:
                self._show_error(str(error))
                return
            self._classification_delete_finished(path)
            return
        self._set_busy(
            True,
            (
                "Dang loai anh khoi data..."
                if self.mode == "simple_class"
                else "Dang archive anh va cap nhat class_f..."
            ),
            delayed_overlay=True,
        )
        task = FunctionTask(
            self.classification_manager.delete_sample,
            sample,
        )
        task.signals.succeeded.connect(
            lambda _, value=path: self._classification_delete_finished(value)
        )
        task.signals.failed.connect(self._background_failed)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._start_task(task)

    def _classification_delete_finished(self, path: Path) -> None:
        self._remove_classification_path_from_lists(path)
        self.image_cache.pop(str(path), None)
        self._refresh_simple_class_counts()
        self.current_index = min(
            self.current_index,
            max(0, len(self.images) - 1),
        )
        self.statusBar().showMessage(
            (
                "Da danh dau loai khoi ban moi; anh nguon van duoc giu"
                if self._staged_simple_manager() is not None
                else "Da loai anh khoi data; Ctrl+Z de phuc hoi"
                if self.mode == "simple_class"
                else "Da xoa anh vao .cvat_nhai_classification_archive"
            ),
            5000,
        )
        self.show_current()
        if self.mode == "simple_class":
            self._sync_simple_deferred_ui()

    def remove_active_box(self) -> None:
        if self.mode != "edit" or self.busy:
            return
        if self.canvas.remove_active_annotation():
            self._update_object_status()
            self.statusBar().showMessage(
                "Da xoa box trong bo nho; Enter de ap dung, F de phuc hoi",
                4000,
            )

    def export_simple_deferred_dataset(self) -> None:
        manager = self._staged_simple_manager()
        if self.mode != "simple_class" or manager is None or self.busy:
            return
        if self._classification_is_dirty():
            self.statusBar().showMessage(
                "Class hien tai chua xep: nhan Enter hoac F truoc khi tao data",
                5000,
            )
            return
        if not manager.has_unexported_changes:
            self.statusBar().showMessage(
                "Khong co thay doi moi de xuat",
                3500,
            )
            return
        self.apply_simple_deferred_destination()
        destination = self.simple_deferred_destination
        if destination is None:
            return
        self._set_busy(
            True,
            "Dang tao dataset moi; dataset nguon duoc giu nguyen...",
        )
        task = FunctionTask(
            manager.export_dataset,
            destination,
            report_progress=True,
        )
        self.export_task = task
        dialog = ExportProgressDialog(self)
        self.export_progress_dialog = dialog
        dialog.cancel_requested.connect(task.cancel)
        task.signals.progress.connect(dialog.update_progress)
        task.signals.succeeded.connect(self._simple_export_finished)
        task.signals.failed.connect(self._export_failed)
        task.signals.finished.connect(self._export_task_finished)
        self._ensure_window_visible()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QTimer.singleShot(
            0,
            lambda value=task: self._start_task(value),
        )

    def _simple_export_finished(self, result: object) -> None:
        if not isinstance(result, SimpleClassificationExportReport):
            return
        manager = self._staged_simple_manager()
        if manager is None:
            return
        try:
            manager.mark_exported(result.revision)
        except Exception as error:
            self._show_error(str(error))
            return
        if self.export_progress_dialog is not None:
            self.export_progress_dialog.finish()
        self.simple_deferred_destination = None
        self.simple_destination_edit.clear()
        self._sync_simple_deferred_ui()
        QMessageBox.information(
            self,
            "Da tao dataset moi",
            (
                "Da sao chep {} anh vao dataset moi.\n"
                "Doi class: {} | Loai bo: {}\n"
                "Dataset nguon khong bi thay doi.\n\n{}"
            ).format(
                result.images,
                result.changed,
                result.deleted,
                result.destination,
            ),
        )
        self.statusBar().showMessage(
            "Da tao dataset moi tai {}".format(result.destination),
            7000,
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
        preview_dialog = ExportSettingsDialog(
            self._load_export_preview_items(),
            self.export_crop_padding,
            self,
        )
        preview_result = preview_dialog.exec()
        if preview_result != QDialog.DialogCode.Accepted:
            preview_dialog.deleteLater()
            return
        self.export_crop_padding = preview_dialog.crop_padding
        self.settings.setValue(
            "datasets/export_crop_padding",
            self.export_crop_padding,
        )
        preview_dialog.deleteLater()
        QTimer.singleShot(0, self._show_export_destination_dialog)

    def _show_export_destination_dialog(self) -> None:
        if self.mode != "edit" or self.editor_index is None or self.busy:
            return
        self._ensure_window_visible()
        dialog = QFileDialog(
            self,
            "Chon thu muc rong de tao yolo_f + class_f",
            str(self.editor_index.root.parent),
        )
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.accepted.connect(
            lambda value=dialog: self._export_destination_selected(value)
        )
        dialog.rejected.connect(self._export_destination_cancelled)
        dialog.destroyed.connect(
            lambda _object=None, value=dialog: (
                self._export_destination_destroyed(value)
            )
        )
        self.export_destination_dialog = dialog
        dialog.open()
        QTimer.singleShot(
            0,
            lambda value=dialog: self._raise_export_destination_dialog(value),
        )

    def _raise_export_destination_dialog(self, dialog: QFileDialog) -> None:
        if self.export_destination_dialog is not dialog:
            return
        dialog.raise_()
        dialog.activateWindow()

    def _export_destination_selected(self, dialog: QFileDialog) -> None:
        selected = dialog.selectedFiles()
        self.export_destination_dialog = None
        if not selected:
            return
        path = Path(selected[0])
        QTimer.singleShot(
            0,
            lambda value=path: self._start_editor_export(value),
        )

    def _export_destination_cancelled(self) -> None:
        self.export_destination_dialog = None
        self._ensure_window_visible()

    def _export_destination_destroyed(
        self,
        dialog: QFileDialog,
    ) -> None:
        if self.export_destination_dialog is dialog:
            self.export_destination_dialog = None

    def _start_editor_export(self, path: Path) -> None:
        if self.mode != "edit" or self.editor_index is None or self.busy:
            return
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
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
            self.export_crop_padding,
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
        self._ensure_window_visible()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        QTimer.singleShot(
            0,
            lambda value=task: self._start_task(value),
        )

    def _load_export_preview_items(
        self,
        maximum: int = 12,
    ) -> List[ExportPreviewItem]:
        if self.editor_index is None or maximum <= 0:
            return []
        samples = list(self.editor_index.samples)
        random.SystemRandom().shuffle(samples)
        if self.current_path is not None:
            samples.sort(
                key=lambda sample: sample.image_path != self.current_path
            )

        result = []
        inspection_limit = max(200, maximum * 25)
        for sample in samples[:inspection_limit]:
            if len(result) >= maximum:
                break
            if not sample.image_path.exists():
                continue
            try:
                with Image.open(sample.image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                annotations = read_yolo_annotations(
                    sample.label_path,
                    image.width,
                    image.height,
                    len(self.editor_index.class_names),
                )
            except (OSError, YoloEditorError):
                continue
            if not annotations:
                continue
            annotation = random.SystemRandom().choice(annotations)
            result.append(
                ExportPreviewItem(
                    image=image,
                    bbox=annotation.bbox,
                    image_name=sample.image_path.name,
                    class_name=self.editor_index.class_names[
                        annotation.class_id
                    ],
                )
            )
        return result

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
        if self._current_edit_is_dirty():
            self.reset_annotation()
            self.statusBar().showMessage(
                "Da hoan tac thay doi chua luu cua anh hien tai",
                3500,
            )
            return
        if self.mode == "edit":
            if self.editor_manager is None:
                return
            function = self.editor_manager.undo_latest
            callback = self._editor_undo_finished
        elif self.mode in self.CLASSIFICATION_MODES:
            if self.classification_manager is None:
                return
            staged_manager = self._staged_simple_manager()
            if staged_manager is not None:
                try:
                    result = staged_manager.undo_latest()
                except Exception as error:
                    self._show_error(str(error))
                    return
                self._classification_undo_finished(result)
                self._sync_simple_deferred_ui()
                return
            function = self.classification_manager.undo_latest
            callback = self._classification_undo_finished
        else:
            function = self.manager.undo_latest
            callback = self._undo_finished
        self._set_busy(
            True,
            "Dang hoan tac...",
            delayed_overlay=self.mode in (
                "edit",
                *self.CLASSIFICATION_MODES,
            ),
        )
        task = FunctionTask(function)
        task.signals.succeeded.connect(callback)
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

    def _editor_undo_finished(self, result: object) -> None:
        if not isinstance(result, YoloUndoResult):
            self.statusBar().showMessage(
                "Khong co thao tac sua YOLO de hoan tac",
                3000,
            )
            return
        sample = result.sample
        path = sample.image_path
        self.editor_samples_by_path[path] = sample
        if path not in self.editor_all_images:
            self.editor_all_images.append(path)
            self.editor_all_images.sort(
                key=lambda value: str(value).casefold()
            )
        self.editor_image_classes[path] = frozenset(
            annotation.class_id for annotation in result.annotations
        )
        self.statusBar().showMessage(
            (
                "Da hoan tac xoa va phuc hoi anh + label YOLO"
                if result.action == "delete_yolo"
                else "Da hoan tac lan sua label YOLO gan nhat"
            ),
            4500,
        )
        if self.mode == "edit":
            self._apply_editor_filters(path, fallback_index=self.current_index)

    def _classification_undo_finished(self, result: object) -> None:
        if not isinstance(result, ClassificationUndoResult):
            self.statusBar().showMessage(
                "Khong co thao tac class_f de hoan tac",
                3000,
            )
            return
        sample = result.sample
        if result.action == "change_classification":
            replaced_path = result.replaced_path
            if replaced_path is not None:
                self._replace_classification_sample(replaced_path, sample)
        else:
            self.classification_samples_by_path[sample.image_path] = sample
            if sample.image_path not in self.classification_all_images:
                self.classification_all_images.append(sample.image_path)
                self.classification_all_images.sort(
                    key=self._classification_path_sort_key
                )
        self.statusBar().showMessage(
            (
                (
                    "Da hoan tac danh dau loai; anh nguon khong doi"
                    if self._staged_simple_manager() is not None
                    else "Da hoan tac xoa va phuc hoi anh"
                    if self.mode == "simple_class"
                    else "Da hoan tac xoa va phuc hoi anh class_f"
                )
                if result.action == "delete_classification"
                else (
                    "Da hoan tac doi class trong phien staging"
                    if self._staged_simple_manager() is not None
                    else "Da hoan tac lan doi class gan nhat"
                )
            ),
            4500,
        )
        if self.mode in self.CLASSIFICATION_MODES:
            self._refresh_simple_class_counts()
            self._apply_classification_filters(
                sample.image_path,
                fallback_index=self.current_index,
            )
            if self.mode == "simple_class":
                self._sync_simple_deferred_ui()

    def navigate(self, delta: int) -> bool:
        if self.busy or not self.images:
            return False
        if self._current_edit_is_dirty():
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
        if (
            self.mode not in ("edit", *self.CLASSIFICATION_MODES)
            or self.busy
            or not self.images
        ):
            return
        target = max(0, min(int(index), len(self.images) - 1))
        if self._current_edit_is_dirty():
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
        if self.mode in ("edit", *self.CLASSIFICATION_MODES) and self.images:
            self.position_label.setText(
                "{} / {}".format(
                    max(0, min(index, len(self.images) - 1)) + 1,
                    len(self.images),
                )
            )

    def _set_busy(
        self,
        busy: bool,
        message: str = "",
        delayed_overlay: bool = False,
    ) -> None:
        self.busy = busy
        if busy:
            self._stop_continuous_navigation()
        self.commit_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.undo_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        self.source_edit.setEnabled(not busy)
        self.source_browse.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.simple_deferred_toggle.setEnabled(not busy)
        self.simple_destination_edit.setEnabled(not busy)
        self.simple_destination_browse.setEnabled(not busy)
        self.simple_publish_button.setEnabled(not busy)
        self.remove_box_button.setEnabled(not busy)
        self.editor_split_filter_combo.setEnabled(not busy)
        self.editor_class_filter_combo.setEnabled(not busy)
        self.canvas.setEnabled(not busy)
        for button in self.class_buttons:
            button.setEnabled(not busy)
        if message:
            self.statusBar().showMessage(message)
        if busy:
            self._ensure_window_visible()
            self.busy_overlay_message = message
            self.busy_overlay_timer.stop()
            if delayed_overlay:
                self.busy_overlay_timer.start(180)
            else:
                self.busy_overlay.show_message(message)
        else:
            self.busy_overlay_timer.stop()
            self.busy_overlay_message = ""
            self.busy_overlay.hide()
        self._sync_simple_deferred_ui()

    def _show_delayed_busy_overlay(self) -> None:
        if self.busy:
            self.busy_overlay.show_message(self.busy_overlay_message)

    def _set_editor_split_filter_ui(self, split: str) -> None:
        if split not in ("all", *SPLITS):
            split = "all"
        combo = getattr(self, "editor_split_filter_combo", None)
        if combo is None:
            self.editor_split_filter = split
            return
        index = combo.findData(split)
        if index < 0:
            index = combo.findData("all")
            split = "all"
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self.editor_split_filter = split

    def _set_editor_class_filter_ui(self, class_filter) -> None:
        if class_filter != "all":
            try:
                class_filter = int(class_filter)
            except (TypeError, ValueError):
                class_filter = "all"
        combo = getattr(self, "editor_class_filter_combo", None)
        if combo is None:
            self.editor_class_filter = class_filter
            return
        index = combo.findData(class_filter)
        if index < 0:
            index = combo.findData("all")
            class_filter = "all"
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self.editor_class_filter = class_filter

    def _editor_split_filter_changed(self) -> None:
        if self.mode not in ("edit", "class_edit"):
            return
        requested = str(
            self.editor_split_filter_combo.currentData() or "all"
        )
        if requested == self.editor_split_filter:
            return
        if self.busy:
            self._set_editor_split_filter_ui(self.editor_split_filter)
            return
        if self._current_edit_is_dirty():
            self._set_editor_split_filter_ui(self.editor_split_filter)
            self.statusBar().showMessage(
                "Nhan Enter de luu hoac F de bo thay doi truoc khi loc split",
                5000,
            )
            return
        self.editor_split_filter = requested
        if self.mode == "edit":
            self._apply_editor_filters(self.current_path)
        else:
            self._apply_classification_filters(self.current_path)

    def _editor_class_filter_changed(self) -> None:
        if self.mode not in ("edit", *self.CLASSIFICATION_MODES):
            return
        requested = self.editor_class_filter_combo.currentData()
        if requested != "all":
            try:
                requested = int(requested)
            except (TypeError, ValueError):
                requested = "all"
        if requested == self.editor_class_filter:
            return
        if self.busy:
            self._set_editor_class_filter_ui(self.editor_class_filter)
            return
        if self._current_edit_is_dirty():
            self._set_editor_class_filter_ui(self.editor_class_filter)
            self.statusBar().showMessage(
                "Nhan Enter de luu hoac F de bo thay doi truoc khi loc class",
                5000,
            )
            return
        self.editor_class_filter = requested
        if self.mode == "edit":
            self._apply_editor_filters(self.current_path)
        else:
            self._apply_classification_filters(self.current_path)

    def _split_title(self, split: str) -> str:
        if split == "all":
            return "tat ca"
        return split.upper()

    def _class_filter_title(self, class_filter) -> str:
        if class_filter == "all":
            return "tat ca class"
        class_id = int(class_filter)
        if 0 <= class_id < len(self.active_class_names):
            return "class {} ({})".format(
                class_id + 1,
                self.active_class_names[class_id],
            )
        return "class {}".format(class_id + 1)

    def _editor_filter_title(self) -> str:
        parts = []
        if self.editor_split_filter != "all":
            parts.append(self._split_title(self.editor_split_filter))
        if self.editor_class_filter != "all":
            parts.append(self._class_filter_title(self.editor_class_filter))
        return " + ".join(parts) if parts else "tat ca"

    def _editor_filter_images(self) -> List[Path]:
        result = []
        for path in self.editor_all_images:
            sample = self.editor_samples_by_path.get(path)
            if sample is None:
                continue
            if (
                self.editor_split_filter != "all"
                and sample.split != self.editor_split_filter
            ):
                continue
            if self.editor_class_filter != "all":
                class_id = int(self.editor_class_filter)
                if class_id not in self.editor_image_classes.get(
                    path,
                    frozenset(),
                ):
                    continue
            result.append(path)
        return result

    def _editor_queue_text(self) -> str:
        if (
            self.editor_split_filter == "all"
            and self.editor_class_filter == "all"
        ):
            return "{} anh dataset".format(len(self.images))
        return "{} anh {} / {} tong".format(
            len(self.images),
            self._editor_filter_title(),
            len(self.editor_all_images),
        )

    def _update_editor_split_status(self) -> None:
        if self.mode != "edit":
            return
        sample = self.editor_samples_by_path.get(self.current_path)
        if sample is None:
            self.editor_split_status.setText("Split hien tai: -")
            return
        self.editor_split_status.setText(
            "Split hien tai: {} | Bo loc: {}".format(
                sample.split.upper(),
                self._editor_filter_title(),
            )
        )

    def _remove_editor_path_from_lists(self, path: Path) -> None:
        self.editor_samples_by_path.pop(path, None)
        self.editor_all_images = [
            value for value in self.editor_all_images if value != path
        ]
        self.images = [value for value in self.images if value != path]
        self.editor_image_classes.pop(path, None)

    def _classification_path_sort_key(self, path: Path):
        sample = self.classification_samples_by_path.get(path)
        if sample is None:
            return len(SPLITS), len(self.active_class_names), str(path).casefold()
        if self._staged_simple_manager() is not None:
            return 0, 0, str(path).casefold()
        split_index = (
            SPLITS.index(sample.split)
            if sample.split in SPLITS
            else 0
        )
        return (
            split_index,
            sample.class_id,
            str(path).casefold(),
        )

    def _classification_filter_images(self) -> List[Path]:
        result = []
        for path in self.classification_all_images:
            sample = self.classification_samples_by_path.get(path)
            if sample is None:
                continue
            if (
                self.editor_split_filter != "all"
                and sample.split != self.editor_split_filter
            ):
                continue
            if (
                self.editor_class_filter != "all"
                and sample.class_id != int(self.editor_class_filter)
            ):
                continue
            result.append(path)
        return result

    def _classification_queue_text(self) -> str:
        if (
            self.editor_split_filter == "all"
            and self.editor_class_filter == "all"
        ):
            if self.mode == "simple_class":
                return "{} anh data don gian | {} class".format(
                    len(self.images),
                    len(self.active_class_names),
                )
            return "{} anh class_f".format(len(self.images))
        suffix = " (data don gian)" if self.mode == "simple_class" else ""
        return "{} anh {} / {} tong{}".format(
            len(self.images),
            self._editor_filter_title(),
            len(self.classification_all_images),
            suffix,
        )

    def _replace_classification_sample(
        self,
        old_path: Path,
        sample: ClassificationSample,
    ) -> None:
        self.classification_samples_by_path.pop(old_path, None)
        self.classification_samples_by_path[sample.image_path] = sample
        self.classification_all_images = [
            sample.image_path if value == old_path else value
            for value in self.classification_all_images
        ]
        self.classification_all_images.sort(
            key=self._classification_path_sort_key
        )
        self.images = [
            sample.image_path if value == old_path else value
            for value in self.images
        ]
        cached = self.image_cache.pop(str(old_path), None)
        if cached is not None:
            self.image_cache[str(sample.image_path)] = cached
        self.classification_original_class_id = sample.class_id

    def _remove_classification_path_from_lists(self, path: Path) -> None:
        self.classification_samples_by_path.pop(path, None)
        self.classification_all_images = [
            value for value in self.classification_all_images if value != path
        ]
        self.images = [value for value in self.images if value != path]

    def _apply_classification_filters(
        self,
        preferred_path: Optional[Path] = None,
        fallback_index: int = 0,
    ) -> None:
        if self.mode not in self.CLASSIFICATION_MODES:
            return
        self.images = self._classification_filter_images()
        if not self.images:
            self.current_index = 0
            self.current_path = None
            self.canvas.clear_image()
            self.classification_original_class_id = -1
            self.file_label.setText(
                "Khong co anh trong bo loc {}".format(
                    self._editor_filter_title()
                )
            )
            self.path_label.clear()
            self.position_label.setText("0 / 0")
            self.queue_label.setText(self._classification_queue_text())
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            self._update_classification_status()
            self.statusBar().showMessage(
                "Bo loc {} khong co anh".format(self._editor_filter_title()),
                4000,
            )
            return
        if preferred_path in self.images:
            self.current_index = self.images.index(preferred_path)
        else:
            self.current_index = min(
                max(0, fallback_index),
                len(self.images) - 1,
            )
        self.show_current()

    def _apply_editor_filters(
        self,
        preferred_path: Optional[Path] = None,
        fallback_index: int = 0,
    ) -> None:
        if self.mode != "edit":
            return
        self.images = self._editor_filter_images()
        if not self.images:
            self.current_index = 0
            self.current_path = None
            self.canvas.clear_image()
            self.file_label.setText(
                "Khong co anh trong bo loc {}".format(
                    self._editor_filter_title()
                )
            )
            self.path_label.clear()
            self.position_label.setText("0 / 0")
            self.queue_label.setText(self._editor_queue_text())
            self.progress.setRange(0, 0)
            self.progress.setValue(0)
            self.editor_original_annotations = ()
            self._update_editor_split_status()
            self.statusBar().showMessage(
                "Bo loc {} khong co anh".format(self._editor_filter_title()),
                4000,
            )
            return
        if preferred_path in self.images:
            self.current_index = self.images.index(preferred_path)
        else:
            self.current_index = min(
                max(0, fallback_index),
                len(self.images) - 1,
            )
        self.show_current()

    def _ensure_window_visible(self) -> None:
        if self.isMinimized():
            self.showNormal()
        elif not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

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
        if self.mode == "simple_class":
            staged = self.simple_deferred_toggle.isChecked()
            if self.classification_index is None:
                self.schema_badge.setText(
                    "Chua mo data don gian{}".format(
                        " - luu ban sao" if staged else ""
                    )
                )
                self.schema_badge.setStyleSheet(
                    "background:#1E293B;color:#CBD5E1;"
                )
            else:
                self.schema_badge.setText(
                    "{} class - data don gian{}".format(
                        len(self.classification_index.class_names),
                        " - staging" if staged else "",
                    )
                )
                self.schema_badge.setStyleSheet(
                    "background:#164E63;color:#A5F3FC;"
                )
            self.migrate_button.setVisible(False)
            return
        if self.mode == "class_edit":
            if self.classification_index is None:
                self.schema_badge.setText("Chua mo class_f")
                self.schema_badge.setStyleSheet(
                    "background:#1E293B;color:#CBD5E1;"
                )
            else:
                self.schema_badge.setText(
                    "{} lop - sua class_f".format(
                        len(self.classification_index.class_names)
                    )
                )
                self.schema_badge.setStyleSheet(
                    "background:#312E81;color:#C4B5FD;"
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
        if self.closing:
            return
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
        if self.busy:
            self.statusBar().showMessage(
                "Dang hoan tat giao dich du lieu; vui long doi trong giay lat",
                3000,
            )
            event.ignore()
            return
        if (
            self._simple_session_has_unsaved_changes()
            and not self._confirm_discard_simple_staging("dong ung dung")
        ):
            event.ignore()
            return
        if self.export_destination_dialog is not None:
            self.export_destination_dialog.close()
            self.export_destination_dialog = None
        self.closing = True
        self._stop_continuous_navigation()
        self.settings.sync()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        overlay = getattr(self, "busy_overlay", None)
        if overlay is not None:
            overlay.setGeometry(self.centralWidget().rect())

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
            self.undo_latest()
            event.accept()
            return
        if Qt.Key_1 <= key <= Qt.Key_9:
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
