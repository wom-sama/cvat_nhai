from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def normalized(self) -> "BBox":
        return BBox(
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    def clamp(self, image_width: int, image_height: int) -> "BBox":
        box = self.normalized()
        return BBox(
            max(0.0, min(float(image_width), box.x1)),
            max(0.0, min(float(image_height), box.y1)),
            max(0.0, min(float(image_width), box.x2)),
            max(0.0, min(float(image_height), box.y2)),
        )

    def is_valid(self, min_size: float = 3.0) -> bool:
        return self.width >= min_size and self.height >= min_size

    def to_yolo(
        self,
        image_width: int,
        image_height: int,
        min_size: float = 3.0,
        clamp_box: bool = True,
    ) -> Tuple[float, float, float, float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        box = (
            self.clamp(image_width, image_height)
            if clamp_box
            else self.normalized()
        )
        if not box.is_valid(min_size=min_size):
            raise ValueError("Bounding box is too small")
        center_x = (box.x1 + box.x2) / 2.0 / image_width
        center_y = (box.y1 + box.y2) / 2.0 / image_height
        width = box.width / image_width
        height = box.height / image_height
        return center_x, center_y, width, height

    def padded(self, ratio: float, image_width: int, image_height: int) -> "BBox":
        box = self.clamp(image_width, image_height)
        pad_x = box.width * max(0.0, ratio)
        pad_y = box.height * max(0.0, ratio)
        return BBox(
            box.x1 - pad_x,
            box.y1 - pad_y,
            box.x2 + pad_x,
            box.y2 + pad_y,
        ).clamp(image_width, image_height)

    @classmethod
    def from_yolo(
        cls,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        image_width: int,
        image_height: int,
        clamp_box: bool = True,
    ) -> "BBox":
        pixel_width = width * image_width
        pixel_height = height * image_height
        pixel_center_x = center_x * image_width
        pixel_center_y = center_y * image_height
        box = cls(
            pixel_center_x - pixel_width / 2.0,
            pixel_center_y - pixel_height / 2.0,
            pixel_center_x + pixel_width / 2.0,
            pixel_center_y + pixel_height / 2.0,
        )
        return box.clamp(image_width, image_height) if clamp_box else box


@dataclass(frozen=True)
class YoloAnnotation:
    class_id: int
    bbox: BBox


@dataclass(frozen=True)
class YoloSample:
    image_path: Path
    label_path: Path
    split: str


@dataclass(frozen=True)
class YoloDatasetIndex:
    root: Path
    data_yaml: Path
    class_names: Tuple[str, ...]
    samples: Tuple[YoloSample, ...]


@dataclass(frozen=True)
class ExportReport:
    destination: Path
    images: int
    objects: int
    skipped: int
    class_counts: Dict[int, int]


@dataclass(frozen=True)
class DatasetPaths:
    detection_root: Path
    classification_root: Path
    archive_root: Path


@dataclass(frozen=True)
class OperationResult:
    operation_id: str
    action: str
    source: Path
    split: Optional[str] = None
    detection_image: Optional[Path] = None
    detection_label: Optional[Path] = None
    classification_crop: Optional[Path] = None
    archived_source: Optional[Path] = None


@dataclass(frozen=True)
class SchemaAudit:
    ready: bool
    detection_names: Tuple[str, ...]
    classification_names: Tuple[str, ...]
    messages: Tuple[str, ...]


@dataclass(frozen=True)
class MigrationReport:
    total_labels: int
    total_objects: int
    changed_labels: int
    unresolved_labels: int
    mismatched_objects: int
    new_class_counts: Dict[int, int]
    issues: Tuple[str, ...]
    backup_path: Optional[Path] = None
