import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageOps

from .constants import CLASS_NAMES, IMAGE_EXTENSIONS, SPLITS
from .journal import OperationJournal
from .models import BBox, DatasetPaths, OperationResult
from .schema import audit_schema, balance_payload, ensure_directory_layout
from .utils import (
    atomic_write_json,
    atomic_write_yaml,
    load_yaml,
    path_is_within,
    safe_stem,
    stable_suffix,
)


class DatasetError(RuntimeError):
    pass


class DatasetManager:
    def __init__(
        self,
        paths: DatasetPaths,
        crop_padding: float = 0.08,
    ) -> None:
        self.paths = paths
        self.crop_padding = max(0.0, float(crop_padding))
        self._lock = threading.Lock()
        self.paths.archive_root.mkdir(parents=True, exist_ok=True)
        self.journal = OperationJournal(
            self.paths.archive_root / "operations.jsonl"
        )

    def validate_ready(self) -> None:
        audit = audit_schema(
            self.paths.detection_root,
            self.paths.classification_root,
        )
        if not audit.ready:
            raise DatasetError(" ".join(audit.messages))

    def choose_split(self, source: Path, requested: str = "auto") -> str:
        if requested in SPLITS:
            return requested
        for part in reversed(source.parts):
            lowered = part.lower()
            if lowered in SPLITS:
                return lowered
        digest = hashlib.sha1(
            str(source.resolve()).casefold().encode("utf-8", errors="replace")
        ).digest()
        bucket = int.from_bytes(digest[:4], "big") / float(2 ** 32)
        if bucket < 0.7:
            return "train"
        if bucket < 0.9:
            return "val"
        return "test"

    def _destination_paths(
        self,
        source: Path,
        split: str,
        class_id: int,
    ) -> Tuple[Path, Path, Path]:
        base_stem = safe_stem(source.stem)
        extension = source.suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            extension = ".jpg"

        image_dir = self.paths.detection_root / "images" / split
        label_dir = self.paths.detection_root / "labels" / split
        crop_dir = (
            self.paths.classification_root
            / split
            / CLASS_NAMES[class_id]
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

        stem = base_stem
        image_path = image_dir / (stem + extension)
        label_path = label_dir / (stem + ".txt")
        crop_path = crop_dir / (stem + "_box000.jpg")
        if image_path.exists() or label_path.exists() or crop_path.exists():
            stem = "{}__{}".format(base_stem, stable_suffix(source))
            image_path = image_dir / (stem + extension)
            label_path = label_dir / (stem + ".txt")
            crop_path = crop_dir / (stem + "_box000.jpg")
            counter = 2
            while image_path.exists() or label_path.exists() or crop_path.exists():
                candidate = "{}_{}".format(stem, counter)
                image_path = image_dir / (candidate + extension)
                label_path = label_dir / (candidate + ".txt")
                crop_path = crop_dir / (candidate + "_box000.jpg")
                counter += 1
        return image_path, label_path, crop_path

    def _archive_path(self, source: Path, operation_id: str) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        target_dir = self.paths.archive_root / "removed" / day / operation_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / source.name

    def _move_to_archive(self, source: Path, operation_id: str) -> Path:
        archive_path = self._archive_path(source, operation_id)
        shutil.move(str(source), str(archive_path))
        return archive_path

    def _append_manifest(self, row: list) -> int:
        path = self.paths.classification_root / "manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        previous_size = path.stat().st_size if existed else 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not existed or previous_size == 0:
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
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        return previous_size

    def _truncate_manifest(self, previous_size: int) -> None:
        path = self.paths.classification_root / "manifest.csv"
        if not path.exists():
            return
        with path.open("r+b") as handle:
            handle.truncate(previous_size)

    def _read_detection_counts(self) -> Tuple[Counter, int]:
        counts = Counter()
        image_count = 0
        for split in SPLITS:
            image_dir = self.paths.detection_root / "images" / split
            if image_dir.exists():
                image_count += sum(
                    1
                    for item in image_dir.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                )
            label_dir = self.paths.detection_root / "labels" / split
            if not label_dir.exists():
                continue
            for label in label_dir.glob("*.txt"):
                try:
                    for line in label.read_text(
                        encoding="utf-8-sig"
                    ).splitlines():
                        parts = line.split()
                        if parts:
                            counts[int(float(parts[0]))] += 1
                except (OSError, ValueError):
                    continue
        return counts, image_count

    def _read_classification_counts(self) -> Tuple[Counter, Dict[str, Counter]]:
        total = Counter()
        per_split = {}
        for split in SPLITS:
            split_counts = Counter()
            for class_id, class_name in enumerate(CLASS_NAMES):
                folder = self.paths.classification_root / split / class_name
                count = 0
                if folder.exists():
                    count = sum(
                        1
                        for item in folder.iterdir()
                        if item.is_file()
                        and item.suffix.lower() in IMAGE_EXTENSIONS
                    )
                split_counts[class_name] = count
                total[class_id] += count
            per_split[split] = split_counts
        return total, per_split

    def refresh_statistics(self) -> None:
        detection_counts, image_count = self._read_detection_counts()
        atomic_write_yaml(
            self.paths.detection_root / "canbang.yaml",
            balance_payload(
                detection_counts,
                image_count,
                "Five-class schema maintained by CVAT Nhai",
            ),
        )

        classification_counts, per_split = self._read_classification_counts()
        total_cls_images = sum(classification_counts.values())
        atomic_write_yaml(
            self.paths.classification_root / "canbang.yaml",
            balance_payload(
                classification_counts,
                total_cls_images,
                "Five-class crop dataset maintained by CVAT Nhai",
            ),
        )
        stats = {
            "mode": "crop-box",
            "base_padding": self.crop_padding,
            "splits": {
                split: {
                    "classes": dict(per_split[split]),
                    "skipped": {},
                }
                for split in SPLITS
            },
        }
        atomic_write_json(
            self.paths.classification_root / "stats.json",
            stats,
        )

    def _increment_statistics(self, class_id: int, split: str) -> None:
        detection_path = self.paths.detection_root / "canbang.yaml"
        detection = load_yaml(detection_path)
        block = detection.get("dataset_balance")
        if isinstance(block, dict) and len(block.get("classes", {})) == 5:
            classes = block["classes"]
            key = str(class_id)
            classes.setdefault(key, {"count": 0})
            classes[key]["count"] = int(classes[key].get("count", 0)) + 1
            block["total_images"] = int(block.get("total_images", 0)) + 1
            block["total_objects"] = int(block.get("total_objects", 0)) + 1
            total = max(1, int(block["total_objects"]))
            for item in classes.values():
                count = int(item.get("count", 0))
                item["ratio"] = round(count / total, 6)
                item["percent"] = round(count / total * 100.0, 2)
            atomic_write_yaml(detection_path, detection)

        stats_path = self.paths.classification_root / "stats.json"
        stats = {}
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                stats = {}
        splits = stats.setdefault("splits", {})
        split_block = splits.setdefault(split, {"classes": {}, "skipped": {}})
        classes = split_block.setdefault("classes", {})
        name = CLASS_NAMES[class_id]
        classes[name] = int(classes.get(name, 0)) + 1
        stats.setdefault("mode", "crop-box")
        stats["base_padding"] = self.crop_padding
        atomic_write_json(stats_path, stats)

        cls_balance_path = self.paths.classification_root / "canbang.yaml"
        cls_balance = load_yaml(cls_balance_path)
        cls_block = cls_balance.get("dataset_balance")
        if isinstance(cls_block, dict) and len(cls_block.get("classes", {})) == 5:
            classes = cls_block["classes"]
            key = str(class_id)
            classes.setdefault(key, {"count": 0})
            classes[key]["count"] = int(classes[key].get("count", 0)) + 1
            cls_block["total_images"] = int(cls_block.get("total_images", 0)) + 1
            cls_block["total_objects"] = int(cls_block.get("total_objects", 0)) + 1
            total = max(1, int(cls_block["total_objects"]))
            for item in classes.values():
                count = int(item.get("count", 0))
                item["ratio"] = round(count / total, 6)
                item["percent"] = round(count / total * 100.0, 2)
            atomic_write_yaml(cls_balance_path, cls_balance)

    def annotate(
        self,
        source: Path,
        bbox: BBox,
        class_id: int,
        requested_split: str = "auto",
    ) -> OperationResult:
        source = source.resolve()
        if not source.is_file():
            raise DatasetError("Source image no longer exists: {}".format(source))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise DatasetError("Select one of the five classes")
        self.validate_ready()
        ensure_directory_layout(
            self.paths.detection_root,
            self.paths.classification_root,
        )
        if path_is_within(
            source,
            (
                self.paths.detection_root,
                self.paths.classification_root,
                self.paths.archive_root,
            ),
        ):
            raise DatasetError("Source folder must be outside output datasets")

        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            box = bbox.clamp(width, height)
            if not box.is_valid():
                raise DatasetError("Bounding box must be at least 3x3 pixels")

            operation_id = uuid.uuid4().hex
            split = self.choose_split(source, requested_split)
            image_path, label_path, crop_path = self._destination_paths(
                source,
                split,
                class_id,
            )
            archive_path = self._archive_path(source, operation_id)
            yolo = box.to_yolo(width, height)
            padded = box.padded(self.crop_padding, width, height)

            temp_dir = Path(
                tempfile.mkdtemp(
                    prefix=".cvat_nhai_",
                    dir=str(self.paths.archive_root),
                )
            )
            staged_image = temp_dir / image_path.name
            staged_label = temp_dir / label_path.name
            staged_crop = temp_dir / crop_path.name
            manifest_size = None
            finalized = []
            try:
                save_kwargs = {}
                if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                    save_kwargs = {"quality": 95, "subsampling": 0}
                image.save(staged_image, **save_kwargs)
                staged_label.write_text(
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        class_id,
                        yolo[0],
                        yolo[1],
                        yolo[2],
                        yolo[3],
                    ),
                    encoding="utf-8",
                )
                crop = image.crop(
                    (
                        int(round(padded.x1)),
                        int(round(padded.y1)),
                        int(round(padded.x2)),
                        int(round(padded.y2)),
                    )
                )
                crop.save(staged_crop, format="JPEG", quality=95)

                with self._lock:
                    for staged, final in (
                        (staged_image, image_path),
                        (staged_label, label_path),
                        (staged_crop, crop_path),
                    ):
                        final.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(str(staged), str(final))
                        finalized.append(final)

                    manifest_size = self._append_manifest(
                        [
                            split,
                            str(image_path),
                            str(crop_path),
                            class_id,
                            CLASS_NAMES[class_id],
                            0,
                        ]
                    )
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(archive_path))
                    self._increment_statistics(class_id, split)

                result = OperationResult(
                    operation_id=operation_id,
                    action="annotate",
                    source=source,
                    split=split,
                    detection_image=image_path,
                    detection_label=label_path,
                    classification_crop=crop_path,
                    archived_source=archive_path,
                )
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "committed",
                        "action": "annotate",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "source": str(source),
                        "archived_source": str(archive_path),
                        "split": split,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "bbox": [box.x1, box.y1, box.x2, box.y2],
                        "detection_image": str(image_path),
                        "detection_label": str(label_path),
                        "classification_crop": str(crop_path),
                    }
                )
                return result
            except Exception as error:
                if archive_path.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archive_path), str(source))
                if manifest_size is not None:
                    self._truncate_manifest(manifest_size)
                for final in reversed(finalized):
                    try:
                        final.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    self.refresh_statistics()
                except Exception:
                    pass
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "failed",
                        "action": "annotate",
                        "source": str(source),
                        "error": repr(error),
                    }
                )
                raise DatasetError(str(error)) from error
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def reject(self, source: Path) -> OperationResult:
        source = source.resolve()
        if not source.is_file():
            raise DatasetError("Source image no longer exists: {}".format(source))
        if path_is_within(
            source,
            (
                self.paths.detection_root,
                self.paths.classification_root,
                self.paths.archive_root,
            ),
        ):
            raise DatasetError("Refusing to delete an output dataset image")

        operation_id = uuid.uuid4().hex
        with self._lock:
            archive_path = self._move_to_archive(source, operation_id)
        self.journal.append(
            {
                "operation_id": operation_id,
                "status": "committed",
                "action": "delete",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "source": str(source),
                "archived_source": str(archive_path),
            }
        )
        return OperationResult(
            operation_id=operation_id,
            action="delete",
            source=source,
            archived_source=archive_path,
        )

    def undo_latest(self) -> Optional[Path]:
        with self._lock:
            record = self.journal.latest_committed()
            if not record:
                return None
            source = Path(record["source"])
            archived = Path(record["archived_source"])
            if source.exists():
                raise DatasetError("Cannot restore because source path already exists")
            if not archived.exists():
                raise DatasetError("Archived source is missing")

            if record["action"] == "annotate":
                manifest = self.paths.classification_root / "manifest.csv"
                rows = []
                if manifest.exists():
                    with manifest.open(
                        "r",
                        newline="",
                        encoding="utf-8-sig",
                    ) as handle:
                        rows = list(csv.DictReader(handle))
                    target_crop = record.get("classification_crop")
                    rows = [
                        row
                        for row in rows
                        if row.get("output_image") != target_crop
                    ]
                    with manifest.open(
                        "w",
                        newline="",
                        encoding="utf-8",
                    ) as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "split",
                                "source_image",
                                "output_image",
                                "class_id",
                                "class_name",
                                "source",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(rows)
                for key in (
                    "detection_image",
                    "detection_label",
                    "classification_crop",
                ):
                    value = record.get(key)
                    if value:
                        try:
                            Path(value).unlink()
                        except FileNotFoundError:
                            pass

            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(archived), str(source))
            self.refresh_statistics()
            self.journal.append(
                {
                    "operation_id": uuid.uuid4().hex,
                    "status": "committed",
                    "action": "undo",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "target_operation_id": record["operation_id"],
                    "restored_source": str(source),
                }
            )
            return source
