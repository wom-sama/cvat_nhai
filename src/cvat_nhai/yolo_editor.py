import csv
import hashlib
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}

_DERIVED_IMAGE_SUFFIX = re.compile(
    (
        r"(?:[_-](?:box|aug(?:ment(?:ed)?)?|copy|flip(?:ped)?|"
        r"mirror(?:ed)?|rot(?:ate|ated|ation)?|crop|variant|ver|v)"
        r"\d*)+$"
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _ExportSource:
    sample: YoloSample
    annotations: Tuple[YoloAnnotation, ...]
    width: int
    height: int
    family_key: str
    visual_key: Tuple[int, str, bytes]


@dataclass(frozen=True)
class _ExportGroup:
    key: str
    source_indexes: Tuple[int, ...]
    class_counts: Counter

    @property
    def image_count(self) -> int:
        return len(self.source_indexes)


def _split_metadata() -> dict:
    return {
        "split_strategy": "stratified_group",
        "split_ratios": dict(SPLIT_RATIOS),
        "leakage_prevention": {
            "unit": "source_image_group",
            "group_by": [
                "normalized_source_family",
                "perceptual_visual_fingerprint",
            ],
        },
    }


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
        **_split_metadata(),
    }


def _detection_yaml(class_names: Sequence[str]) -> dict:
    return {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
        "names": [str(name) for name in class_names],
        **_split_metadata(),
    }


def _export_balance(
    counts: Dict[int, int],
    class_count: int,
    total_images: int,
    split_class_counts: Dict[str, Counter],
    split_image_counts: Dict[str, int],
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
            **_split_metadata(),
            "classes": classes,
            "splits": {
                split: {
                    "total_images": int(split_image_counts[split]),
                    "total_objects": int(
                        sum(split_class_counts[split].values())
                    ),
                    "classes": {
                        str(class_id): int(
                            split_class_counts[split][class_id]
                        )
                        for class_id in range(class_count)
                    },
                }
                for split in SPLITS
            },
        }
    }


def _source_family(path: Path) -> str:
    stem = safe_stem(path.stem).casefold()
    family = _DERIVED_IMAGE_SUFFIX.sub("", stem).rstrip("._-")
    return family or stem


def _visual_fingerprint(image: Image.Image) -> Tuple[int, str, bytes]:
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    difference_hash = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            difference_hash <<= 1
            difference_hash |= int(
                pixels[offset + column]
                > pixels[offset + column + 1]
            )
    average = image.resize((1, 1), Image.Resampling.BILINEAR).getpixel(
        (0, 0)
    )
    color_key = "".join(str(int(channel) // 64) for channel in average)
    color_thumbnail = image.resize(
        (16, 16),
        Image.Resampling.BILINEAR,
    )
    quantized_thumbnail = bytes(
        value // 32 for value in color_thumbnail.tobytes()
    )
    return difference_hash, color_key, quantized_thumbnail


def _thumbnails_are_near(left: bytes, right: bytes) -> bool:
    maximum_difference = len(left) * 0.4
    total_difference = 0
    for left_value, right_value in zip(left, right):
        total_difference += abs(left_value - right_value)
        if total_difference > maximum_difference:
            return False
    return True


def _load_export_sources(
    index: YoloDatasetIndex,
) -> Tuple[List[_ExportSource], int]:
    sources = []
    skipped = 0
    for sample in index.samples:
        if not sample.image_path.exists():
            skipped += 1
            continue
        with Image.open(sample.image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            annotations = tuple(
                read_yolo_annotations(
                    sample.label_path,
                    width,
                    height,
                    len(index.class_names),
                )
            )
            sources.append(
                _ExportSource(
                    sample=sample,
                    annotations=annotations,
                    width=width,
                    height=height,
                    family_key=_source_family(sample.image_path),
                    visual_key=_visual_fingerprint(image),
                )
            )
    return sources, skipped


def _build_export_groups(
    sources: Sequence[_ExportSource],
) -> List[_ExportGroup]:
    parents = list(range(len(sources)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    family_owners = {}
    visual_owners = {}
    visual_bands: Dict[Tuple[str, int, int], List[int]] = {}
    for index, source in enumerate(sources):
        previous_family = family_owners.get(source.family_key)
        if previous_family is None:
            family_owners[source.family_key] = index
        else:
            union(index, previous_family)

        previous_visual = visual_owners.get(source.visual_key)
        if previous_visual is not None:
            union(index, previous_visual)
            continue
        visual_owners[source.visual_key] = index
        visual_hash, color_key, thumbnail = source.visual_key
        checked_candidates = set()
        for band_index in range(8):
            band_value = (visual_hash >> (band_index * 8)) & 0xFF
            bucket_key = (color_key, band_index, band_value)
            bucket = visual_bands.setdefault(bucket_key, [])
            for candidate_index in bucket:
                if candidate_index in checked_candidates:
                    continue
                checked_candidates.add(candidate_index)
                candidate_hash, _, candidate_thumbnail = sources[
                    candidate_index
                ].visual_key
                if (
                    bin(visual_hash ^ candidate_hash).count("1") <= 3
                    and _thumbnails_are_near(
                        thumbnail,
                        candidate_thumbnail,
                    )
                ):
                    union(index, candidate_index)
            bucket.append(index)

    members: Dict[int, List[int]] = {}
    for index in range(len(sources)):
        members.setdefault(find(index), []).append(index)

    groups = []
    for indexes in members.values():
        ordered = tuple(
            sorted(
                indexes,
                key=lambda value: str(
                    sources[value].sample.image_path
                ).casefold(),
            )
        )
        paths = "\n".join(
            str(sources[index].sample.image_path.resolve()).casefold()
            for index in ordered
        )
        class_counts = Counter()
        for index in ordered:
            class_counts.update(
                annotation.class_id
                for annotation in sources[index].annotations
            )
        groups.append(
            _ExportGroup(
                key=hashlib.sha256(
                    paths.encode("utf-8", errors="replace")
                ).hexdigest()[:16],
                source_indexes=ordered,
                class_counts=class_counts,
            )
        )
    return groups


def _assign_group_splits(
    sources: Sequence[_ExportSource],
    groups: Sequence[_ExportGroup],
    class_count: int,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    total_class_counts = Counter()
    for group in groups:
        total_class_counts.update(group.class_counts)
    total_images = len(sources)
    target_classes = {
        split: {
            class_id: total_class_counts[class_id] * SPLIT_RATIOS[split]
            for class_id in range(class_count)
        }
        for split in SPLITS
    }
    target_images = {
        split: total_images * SPLIT_RATIOS[split]
        for split in SPLITS
    }
    current_classes = {
        split: Counter() for split in SPLITS
    }
    current_images = Counter()

    def priority(group: _ExportGroup) -> tuple:
        rarity = sum(
            count / max(1, total_class_counts[class_id])
            for class_id, count in group.class_counts.items()
        )
        return (
            -rarity,
            -sum(group.class_counts.values()),
            -group.image_count,
            group.key,
        )

    source_splits = {}
    source_groups = {}
    for group in sorted(groups, key=priority):
        scored_splits = []
        for split_order, split in enumerate(SPLITS):
            class_delta = 0.0
            for class_id in range(class_count):
                total = max(1, total_class_counts[class_id])
                before = (
                    current_classes[split][class_id]
                    - target_classes[split][class_id]
                ) / total
                after = (
                    current_classes[split][class_id]
                    + group.class_counts[class_id]
                    - target_classes[split][class_id]
                ) / total
                class_delta += after * after - before * before
            image_total = max(1, total_images)
            image_before = (
                current_images[split] - target_images[split]
            ) / image_total
            image_after = (
                current_images[split]
                + group.image_count
                - target_images[split]
            ) / image_total
            image_delta = image_after * image_after - image_before * image_before
            scored_splits.append(
                (class_delta + 0.35 * image_delta, split_order, split)
            )
        assigned_split = min(scored_splits)[2]
        current_classes[assigned_split].update(group.class_counts)
        current_images[assigned_split] += group.image_count
        for source_index in group.source_indexes:
            source_splits[source_index] = assigned_split
            source_groups[source_index] = group.key
    return source_splits, source_groups


def _unique_output_stems(
    sources: Sequence[_ExportSource],
    source_splits: Dict[int, str],
) -> Dict[int, str]:
    used = {split: set() for split in SPLITS}
    result = {}
    for index, source in sorted(
        enumerate(sources),
        key=lambda item: str(item[1].sample.image_path).casefold(),
    ):
        split = source_splits[index]
        base = safe_stem(source.sample.image_path.stem)
        candidate = base
        suffix = hashlib.sha1(
            str(source.sample.image_path.resolve()).encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()[:8]
        if candidate.casefold() in used[split]:
            candidate = "{}__{}".format(base, suffix)
        counter = 2
        while candidate.casefold() in used[split]:
            candidate = "{}__{}_{:02d}".format(base, suffix, counter)
            counter += 1
        used[split].add(candidate.casefold())
        result[index] = candidate
    return result


def export_rebalanced_datasets(
    index: YoloDatasetIndex,
    destination: Path,
    crop_padding: float = 0.08,
) -> ExportReport:
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(index.root.resolve())
    except ValueError:
        pass
    else:
        raise YoloEditorError(
            "Thu muc export phai nam ngoai dataset YOLO nguon"
        )
    if destination.exists() and any(destination.iterdir()):
        raise YoloEditorError("Thu muc export phai rong")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".cvat_nhai_export_",
            dir=str(destination.parent),
        )
    )
    detection_root = staging / "dataset"
    classification_root = staging / "cls_crops"
    counts = Counter()
    split_class_counts: Dict[str, Counter] = {
        split: Counter() for split in SPLITS
    }
    split_yolo_image_counts = Counter()
    split_classification_image_counts = Counter()
    classification_manifest_rows = []
    detection_manifest_rows = []
    try:
        sources, skipped = _load_export_sources(index)
        groups = _build_export_groups(sources)
        source_splits, source_groups = _assign_group_splits(
            sources,
            groups,
            len(index.class_names),
        )
        output_stems = _unique_output_stems(sources, source_splits)

        for split in SPLITS:
            (detection_root / "images" / split).mkdir(
                parents=True,
                exist_ok=True,
            )
            (detection_root / "labels" / split).mkdir(
                parents=True,
                exist_ok=True,
            )
            for class_name in index.class_names:
                (
                    classification_root
                    / split
                    / safe_stem(class_name)
                ).mkdir(
                    parents=True,
                    exist_ok=True,
                )

        for source_index, source in enumerate(sources):
            sample = source.sample
            split = source_splits[source_index]
            group_key = source_groups[source_index]
            output_stem = output_stems[source_index]
            output_image = (
                detection_root
                / "images"
                / split
                / (output_stem + sample.image_path.suffix.lower())
            )
            output_label = (
                detection_root
                / "labels"
                / split
                / (output_stem + ".txt")
            )
            atomic_write_text(
                output_label,
                serialize_yolo_annotations(
                    source.annotations,
                    source.width,
                    source.height,
                    len(index.class_names),
                ),
            )
            split_yolo_image_counts[split] += 1
            detection_manifest_rows.append(
                [
                    split,
                    sample.split,
                    group_key,
                    str(sample.image_path),
                    str(
                        destination
                        / output_image.relative_to(staging)
                    ),
                    str(
                        destination
                        / output_label.relative_to(staging)
                    ),
                ]
            )

            with Image.open(sample.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                orientation = opened.getexif().get(274, 1)
                if orientation in (None, 1):
                    shutil.copy2(sample.image_path, output_image)
                elif output_image.suffix.lower() in {".jpg", ".jpeg"}:
                    image.save(
                        output_image,
                        format="JPEG",
                        quality=95,
                    )
                elif output_image.suffix.lower() == ".webp":
                    image.save(
                        output_image,
                        format="WEBP",
                        quality=95,
                    )
                else:
                    image.save(output_image)
                if source.annotations:
                    split_classification_image_counts[split] += 1
                for object_index, annotation in enumerate(
                    source.annotations
                ):
                    class_name = safe_stem(
                        index.class_names[annotation.class_id]
                    )
                    output_dir = (
                        classification_root / split / class_name
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_name = "{}_box{:03d}.jpg".format(
                        output_stem,
                        object_index,
                    )
                    output_path = output_dir / output_name
                    box = annotation.bbox.padded(
                        crop_padding,
                        source.width,
                        source.height,
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
                    split_class_counts[split][annotation.class_id] += 1
                    classification_manifest_rows.append(
                        [
                            split,
                            str(sample.image_path),
                            str(
                                destination
                                / output_path.relative_to(staging)
                            ),
                            annotation.class_id,
                            class_name,
                            object_index,
                            sample.split,
                            group_key,
                        ]
                    )

        with (classification_root / "manifest.csv").open(
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
                    "source_split",
                    "leakage_group",
                ]
            )
            writer.writerows(classification_manifest_rows)
        with (detection_root / "manifest.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "split",
                    "source_split",
                    "leakage_group",
                    "source_image",
                    "output_image",
                    "output_label",
                ]
            )
            writer.writerows(detection_manifest_rows)
        atomic_write_yaml(
            detection_root / "data.yaml",
            _detection_yaml(index.class_names),
        )
        atomic_write_yaml(
            classification_root / "data.yaml",
            _classification_yaml(index.class_names),
        )
        detection_balance = _export_balance(
            counts,
            len(index.class_names),
            len(sources),
            split_class_counts,
            split_yolo_image_counts,
        )
        classification_balance = _export_balance(
            counts,
            len(index.class_names),
            sum(bool(source.annotations) for source in sources),
            split_class_counts,
            split_classification_image_counts,
        )
        atomic_write_yaml(
            detection_root / "canbang.yaml",
            detection_balance,
        )
        atomic_write_yaml(
            classification_root / "canbang.yaml",
            classification_balance,
        )
        atomic_write_json(
            classification_root / "stats.json",
            {
                "mode": "crop-box",
                "base_padding": float(crop_padding),
                **_split_metadata(),
                "source_groups": len(groups),
                "splits": {
                    split: {
                        "images": int(
                            split_classification_image_counts[split]
                        ),
                        "classes": {
                            safe_stem(index.class_names[class_id]): int(
                                split_class_counts[split][class_id]
                            )
                            for class_id in range(
                                len(index.class_names)
                            )
                        },
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
            images=len(sources),
            objects=sum(counts.values()),
            skipped=skipped,
            class_counts=dict(counts),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_classification_folder(
    index: YoloDatasetIndex,
    destination: Path,
    crop_padding: float = 0.08,
) -> ExportReport:
    return export_rebalanced_datasets(
        index,
        destination,
        crop_padding,
    )
