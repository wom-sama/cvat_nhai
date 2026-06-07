import csv
import os
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image, ImageOps

from .constants import IMAGE_EXTENSIONS, SPLITS
from .journal import OperationJournal
from .models import (
    BBox,
    ExportReport,
    YoloAnnotation,
    YoloDatasetIndex,
    YoloSample,
)
from .utils import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    load_yaml,
    names_from_yaml,
    safe_stem,
)


class YoloEditorError(RuntimeError):
    pass


def _resolve_data_yaml(source: Path) -> Path:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in {".yaml", ".yml"}:
            raise YoloEditorError("Hay chon data.yaml hoac thu muc dataset YOLO")
        return source
    candidate = source / "data.yaml"
    if candidate.is_file():
        return candidate
    yaml_files = sorted(source.glob("*.yaml")) + sorted(source.glob("*.yml"))
    if len(yaml_files) == 1:
        return yaml_files[0]
    raise YoloEditorError("Khong tim thay data.yaml trong {}".format(source))


def _dataset_root(data_yaml: Path, payload: dict) -> Path:
    configured = payload.get("path")
    if not configured:
        return data_yaml.parent.resolve()
    root = Path(str(configured))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _split_image_dirs(root: Path, payload: dict, split: str) -> List[Path]:
    configured = payload.get(split)
    if not configured:
        return []
    values = configured if isinstance(configured, list) else [configured]
    directories = []
    for value in values:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_file():
            raise YoloEditorError(
                "{} uses an image-list file; directory-based YOLO is required".format(
                    split
                )
            )
        if path.is_dir():
            directories.append(path)
    return directories


def _label_dir_for_images(root: Path, image_dir: Path, split: str) -> Path:
    try:
        relative = image_dir.relative_to(root)
    except ValueError:
        relative = Path()
    parts = list(relative.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return root.joinpath(*parts)
    return root / "labels" / split


def scan_yolo_dataset(source: Path) -> YoloDatasetIndex:
    data_yaml = _resolve_data_yaml(source)
    payload = load_yaml(data_yaml)
    class_names = tuple(names_from_yaml(payload))
    if not class_names:
        raise YoloEditorError("data.yaml khong co names")

    root = _dataset_root(data_yaml, payload)
    samples = []
    seen_images = set()
    for split in SPLITS:
        for image_dir in _split_image_dirs(root, payload, split):
            label_dir = _label_dir_for_images(root, image_dir, split)
            for image_path in image_dir.rglob("*"):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    continue
                key = os.path.normcase(str(image_path))
                if key in seen_images:
                    continue
                seen_images.add(key)
                relative = image_path.relative_to(image_dir)
                label_path = label_dir / relative.with_suffix(".txt")
                samples.append(
                    YoloSample(
                        image_path=image_path,
                        label_path=label_path,
                        split=split,
                    )
                )
    samples.sort(key=lambda sample: str(sample.image_path).casefold())
    if not samples:
        raise YoloEditorError("Dataset khong co anh YOLO")
    return YoloDatasetIndex(
        root=root,
        data_yaml=data_yaml,
        class_names=class_names,
        samples=tuple(samples),
    )


def read_yolo_annotations(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_count: int,
) -> List[YoloAnnotation]:
    if image_width <= 0 or image_height <= 0:
        raise YoloEditorError("Kich thuoc anh khong hop le")
    if not label_path.exists():
        return []

    annotations = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise YoloEditorError(
                "{}:{} khong phai bbox YOLO 5 cot".format(
                    label_path,
                    line_number,
                )
            )
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = (
                float(value) for value in parts[1:5]
            )
        except ValueError as error:
            raise YoloEditorError(
                "{}:{} co gia tri khong hop le".format(
                    label_path,
                    line_number,
                )
            ) from error
        if not 0 <= class_id < class_count:
            raise YoloEditorError(
                "{}:{} class {} nam ngoai data.yaml".format(
                    label_path,
                    line_number,
                    class_id,
                )
            )
        if (
            not 0.0 <= center_x <= 1.0
            or not 0.0 <= center_y <= 1.0
            or not 0.0 < width <= 1.0
            or not 0.0 < height <= 1.0
        ):
            raise YoloEditorError(
                "{}:{} bbox normalized khong hop le".format(
                    label_path,
                    line_number,
                )
            )
        raw_x1 = (center_x - width / 2.0) * image_width
        raw_y1 = (center_y - height / 2.0) * image_height
        raw_x2 = (center_x + width / 2.0) * image_width
        raw_y2 = (center_y + height / 2.0) * image_height
        rounding_tolerance = 0.001
        if (
            raw_x1 < -rounding_tolerance
            or raw_y1 < -rounding_tolerance
            or raw_x2 > image_width + rounding_tolerance
            or raw_y2 > image_height + rounding_tolerance
        ):
            raise YoloEditorError(
                "{}:{} bbox vuot ngoai bien anh".format(
                    label_path,
                    line_number,
                )
            )
        bbox = BBox.from_yolo(
            center_x,
            center_y,
            width,
            height,
            image_width,
            image_height,
            clamp_box=False,
        )
        if not bbox.is_valid(min_size=0.01):
            raise YoloEditorError(
                "{}:{} bbox co kich thuoc bang 0".format(
                    label_path,
                    line_number,
                )
            )
        annotations.append(YoloAnnotation(class_id=class_id, bbox=bbox))
    return annotations


def serialize_yolo_annotations(
    annotations: Sequence[YoloAnnotation],
    image_width: int,
    image_height: int,
    class_count: int,
) -> str:
    lines = []
    for annotation in annotations:
        if not 0 <= annotation.class_id < class_count:
            raise YoloEditorError(
                "Class {} nam ngoai data.yaml".format(annotation.class_id)
            )
        center_x, center_y, width, height = annotation.bbox.to_yolo(
            image_width,
            image_height,
            min_size=0.01,
            clamp_box=False,
        )
        lines.append(
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                annotation.class_id,
                center_x,
                center_y,
                width,
                height,
            )
        )
    return ("\n".join(lines) + "\n") if lines else ""


class YoloDatasetEditor:
    def __init__(self, index: YoloDatasetIndex) -> None:
        self.index = index
        self.archive_root = index.root / ".cvat_nhai_editor_archive"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        self.journal = OperationJournal(self.archive_root / "operations.jsonl")
        self._lock = threading.Lock()

    def _operation_dir(self, category: str, operation_id: str) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        path = self.archive_root / category / day / operation_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _update_balance(
        self,
        old_annotations: Sequence[YoloAnnotation],
        new_annotations: Sequence[YoloAnnotation],
        image_delta: int = 0,
    ) -> None:
        path = self.index.root / "canbang.yaml"
        if not path.exists():
            return
        payload = load_yaml(path)
        block = payload.get("dataset_balance")
        if not isinstance(block, dict):
            return
        classes = block.get("classes")
        if not isinstance(classes, dict):
            return
        old_counts = Counter(item.class_id for item in old_annotations)
        new_counts = Counter(item.class_id for item in new_annotations)
        for class_id in range(len(self.index.class_names)):
            entry = classes.setdefault(str(class_id), {"count": 0})
            entry["count"] = max(
                0,
                int(entry.get("count", 0))
                - old_counts[class_id]
                + new_counts[class_id],
            )
        block["total_images"] = max(
            0,
            int(block.get("total_images", 0)) + image_delta,
        )
        total_objects = sum(
            int(entry.get("count", 0)) for entry in classes.values()
        )
        block["total_objects"] = total_objects
        for entry in classes.values():
            count = int(entry.get("count", 0))
            ratio = count / total_objects if total_objects else 0.0
            entry["ratio"] = round(ratio, 6)
            entry["percent"] = round(ratio * 100.0, 2)
        atomic_write_yaml(path, payload)

    def save_annotations(
        self,
        sample: YoloSample,
        annotations: Sequence[YoloAnnotation],
        image_width: int,
        image_height: int,
    ) -> None:
        if not sample.image_path.exists():
            raise YoloEditorError("Anh khong con ton tai")
        old_annotations = read_yolo_annotations(
            sample.label_path,
            image_width,
            image_height,
            len(self.index.class_names),
        )
        content = serialize_yolo_annotations(
            annotations,
            image_width,
            image_height,
            len(self.index.class_names),
        )
        operation_id = uuid.uuid4().hex
        backup_dir = self._operation_dir("edits", operation_id)
        backup_label = backup_dir / sample.label_path.name
        with self._lock:
            label_existed = sample.label_path.exists()
            if label_existed:
                shutil.copy2(sample.label_path, backup_label)
            else:
                (backup_dir / ".label_was_missing").write_text(
                    "",
                    encoding="utf-8",
                )
            try:
                atomic_write_text(sample.label_path, content)
                self._update_balance(old_annotations, annotations)
            except Exception:
                if label_existed:
                    shutil.copy2(backup_label, sample.label_path)
                else:
                    try:
                        sample.label_path.unlink()
                    except FileNotFoundError:
                        pass
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "failed",
                        "action": "edit_yolo",
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "image": str(sample.image_path),
                        "label": str(sample.label_path),
                    }
                )
                raise
        self.journal.append(
            {
                "operation_id": operation_id,
                "status": "committed",
                "action": "edit_yolo",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image": str(sample.image_path),
                "label": str(sample.label_path),
                "backup_label": str(backup_label),
                "old_objects": len(old_annotations),
                "new_objects": len(annotations),
            }
        )

    def delete_sample(
        self,
        sample: YoloSample,
        image_width: int,
        image_height: int,
    ) -> None:
        if not sample.image_path.exists():
            raise YoloEditorError("Anh khong con ton tai")
        old_annotations = read_yolo_annotations(
            sample.label_path,
            image_width,
            image_height,
            len(self.index.class_names),
        )
        operation_id = uuid.uuid4().hex
        archive_dir = self._operation_dir("deleted", operation_id)
        archived_image = archive_dir / sample.image_path.name
        archived_label = archive_dir / sample.label_path.name
        with self._lock:
            shutil.move(str(sample.image_path), str(archived_image))
            try:
                if sample.label_path.exists():
                    shutil.move(str(sample.label_path), str(archived_label))
                self._update_balance(old_annotations, (), image_delta=-1)
            except Exception:
                sample.image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archived_image), str(sample.image_path))
                if archived_label.exists():
                    sample.label_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archived_label), str(sample.label_path))
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "failed",
                        "action": "delete_yolo",
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "image": str(sample.image_path),
                        "label": str(sample.label_path),
                    }
                )
                raise
        self.journal.append(
            {
                "operation_id": operation_id,
                "status": "committed",
                "action": "delete_yolo",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image": str(sample.image_path),
                "label": str(sample.label_path),
                "archived_image": str(archived_image),
                "archived_label": str(archived_label),
                "objects": len(old_annotations),
            }
        )


def _classification_yaml(class_names: Sequence[str]) -> dict:
    return {
        "format": "classification_folder",
        "path": ".",
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(class_names),
        "class_name_mode": "raw",
        "names": {
            index: str(name) for index, name in enumerate(class_names)
        },
    }


def _classification_balance(
    counts: Dict[int, int],
    class_count: int,
    total_images: int,
) -> dict:
    total_objects = sum(counts.values())
    classes = {}
    for class_id in range(class_count):
        count = int(counts.get(class_id, 0))
        ratio = count / total_objects if total_objects else 0.0
        classes[str(class_id)] = {
            "count": count,
            "ratio": round(ratio, 6),
            "percent": round(ratio * 100.0, 2),
        }
    return {
        "dataset_balance": {
            "version_note": "Exported from edited YOLO dataset by CVAT Nhai",
            "total_images": total_images,
            "total_objects": total_objects,
            "classes": classes,
        }
    }


def export_classification_folder(
    index: YoloDatasetIndex,
    destination: Path,
    crop_padding: float = 0.08,
) -> ExportReport:
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise YoloEditorError("Thu muc export phai rong")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".cvat_nhai_cls_export_",
            dir=str(destination.parent),
        )
    )
    counts = Counter()
    split_counts: Dict[str, Counter] = {
        split: Counter() for split in SPLITS
    }
    manifest_rows = []
    exported_images = set()
    skipped = 0
    try:
        for split in SPLITS:
            for class_name in index.class_names:
                (staging / split / safe_stem(class_name)).mkdir(
                    parents=True,
                    exist_ok=True,
                )
        for sample in index.samples:
            if not sample.image_path.exists():
                skipped += 1
                continue
            with Image.open(sample.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                annotations = read_yolo_annotations(
                    sample.label_path,
                    width,
                    height,
                    len(index.class_names),
                )
                for object_index, annotation in enumerate(annotations):
                    class_name = safe_stem(
                        index.class_names[annotation.class_id]
                    )
                    output_dir = staging / sample.split / class_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_name = "{}_box{:03d}.jpg".format(
                        safe_stem(sample.image_path.stem),
                        object_index,
                    )
                    output_path = output_dir / output_name
                    if output_path.exists():
                        output_name = "{}__{}_box{:03d}.jpg".format(
                            safe_stem(sample.image_path.stem),
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                str(sample.image_path),
                            ).hex[:8],
                            object_index,
                        )
                        output_path = output_dir / output_name
                    box = annotation.bbox.padded(
                        crop_padding,
                        width,
                        height,
                    )
                    crop = image.crop(
                        (
                            int(round(box.x1)),
                            int(round(box.y1)),
                            int(round(box.x2)),
                            int(round(box.y2)),
                        )
                    )
                    crop.save(output_path, format="JPEG", quality=95)
                    counts[annotation.class_id] += 1
                    split_counts[sample.split][class_name] += 1
                    exported_images.add(sample.image_path)
                    manifest_rows.append(
                        [
                            sample.split,
                            str(sample.image_path),
                            str(destination / output_path.relative_to(staging)),
                            annotation.class_id,
                            class_name,
                            object_index,
                        ]
                    )

        with (staging / "manifest.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "split",
                    "source_image",
                    "output_image",
                    "class_id",
                    "class_name",
                    "source",
                ]
            )
            writer.writerows(manifest_rows)
        atomic_write_yaml(
            staging / "data.yaml",
            _classification_yaml(index.class_names),
        )
        atomic_write_yaml(
            staging / "canbang.yaml",
            _classification_balance(
                counts,
                len(index.class_names),
                len(exported_images),
            ),
        )
        atomic_write_json(
            staging / "stats.json",
            {
                "mode": "crop-box",
                "base_padding": float(crop_padding),
                "splits": {
                    split: {
                        "classes": dict(split_counts[split]),
                        "skipped": {},
                    }
                    for split in SPLITS
                },
            },
        )
        if destination.exists():
            destination.rmdir()
        os.replace(str(staging), str(destination))
        return ExportReport(
            destination=destination,
            images=len(exported_images),
            objects=sum(counts.values()),
            skipped=skipped,
            class_counts=dict(counts),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
