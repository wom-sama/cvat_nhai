import csv
import os
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import CLASS_NAMES, SPLITS
from .image_ops import CLASSIFICATION_IMAGE_SIZE
from .journal import OperationJournal
from .models import (
    ClassificationDatasetIndex,
    ClassificationSample,
    ClassificationUndoResult,
)
from .scanner import scan_images
from .utils import (
    atomic_write_json,
    atomic_write_yaml,
    load_yaml,
    names_from_yaml,
    safe_stem,
)


class ClassificationEditorError(RuntimeError):
    pass


DEFAULT_MANIFEST_FIELDS = (
    "split",
    "source_image",
    "output_image",
    "class_id",
    "class_name",
    "source",
    "source_split",
    "leakage_group",
)

MANIFEST_PATH_FIELDS = (
    "output_image",
    "classification_crop",
    "crop_path",
    "path",
)


@dataclass
class _ManifestDelta:
    changed: List[Tuple[dict, dict]] = field(default_factory=list)
    appended: List[dict] = field(default_factory=list)
    removed: List[Tuple[int, dict]] = field(default_factory=list)


def _resolve_existing_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _resolve_classification_root(source: Path) -> Path:
    source = _resolve_existing_path(source)
    if not source.is_dir():
        raise ClassificationEditorError(
            "Thu muc class_f khong ton tai: {}".format(source)
        )
    if (source / "data.yaml").exists():
        return source
    nested = source / "class_f"
    if (nested / "data.yaml").exists() or any(
        (nested / split).is_dir() for split in SPLITS
    ):
        return nested
    return source


def _read_class_names(root: Path) -> Tuple[str, ...]:
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        names = tuple(names_from_yaml(load_yaml(data_yaml)))
        if names:
            return names
    return tuple(CLASS_NAMES)


def _class_folder(root: Path, split: str, class_name: str) -> Path:
    return root / split / safe_stem(class_name)


def _candidate_class_folders(
    root: Path,
    split: str,
    class_name: str,
) -> Tuple[Path, ...]:
    folders = [_class_folder(root, split, class_name)]
    raw = root / split / str(class_name)
    if raw not in folders:
        folders.append(raw)
    return tuple(folders)


def scan_classification_dataset(source: Path) -> ClassificationDatasetIndex:
    root = _resolve_classification_root(source)
    class_names = _read_class_names(root)
    samples: List[ClassificationSample] = []
    seen = set()
    for split in SPLITS:
        for class_id, class_name in enumerate(class_names):
            for folder in _candidate_class_folders(root, split, class_name):
                if not folder.is_dir():
                    continue
                for image_path in scan_images(folder):
                    resolved = _resolve_existing_path(image_path)
                    key = str(resolved).casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    samples.append(
                        ClassificationSample(
                            image_path=resolved,
                            split=split,
                            class_id=class_id,
                            class_name=str(class_name),
                        )
                    )
    samples.sort(
        key=lambda sample: (
            SPLITS.index(sample.split),
            sample.class_id,
            str(sample.image_path).casefold(),
        )
    )
    if not samples:
        raise ClassificationEditorError(
            "Khong tim thay anh trong class_f: {}".format(root)
        )
    data_yaml = root / "data.yaml"
    return ClassificationDatasetIndex(
        root=root,
        data_yaml=data_yaml if data_yaml.exists() else None,
        class_names=class_names,
        samples=tuple(samples),
    )


class ClassificationDatasetEditor:
    def __init__(self, index: ClassificationDatasetIndex) -> None:
        self.index = index
        self.archive_root = (
            index.root / ".cvat_nhai_classification_archive"
        )
        self.journal = OperationJournal(
            self.archive_root / "operations.jsonl"
        )
        self._lock = threading.RLock()
        self._manifest_exists = self._manifest_path().exists()
        (
            self._manifest_fieldnames,
            self._manifest_rows,
        ) = self._read_manifest()
        self._manifest_rows_by_path: Dict[str, List[dict]] = {}
        self._rebuild_manifest_index()
        self._manifest_signature = self._file_signature(
            self._manifest_path()
        )
        self._counts = Counter()
        self._split_class_counts = {
            split: Counter() for split in SPLITS
        }
        self._split_image_counts = {split: 0 for split in SPLITS}
        for sample in index.samples:
            self._counts[sample.class_id] += 1
            self._split_class_counts[sample.split][sample.class_id] += 1
            self._split_image_counts[sample.split] += 1

    @property
    def root(self) -> Path:
        return self.index.root

    @property
    def class_names(self) -> Tuple[str, ...]:
        return self.index.class_names

    def change_class(
        self,
        sample: ClassificationSample,
        class_id: int,
    ) -> ClassificationSample:
        with self._lock:
            self._validate_sample(sample)
            self._validate_class_id(class_id)
            if sample.class_id == class_id:
                return sample
            self._assert_manifest_unchanged()

            operation_id = uuid.uuid4().hex
            old_path = sample.image_path
            target_dir = _class_folder(
                self.root,
                sample.split,
                self.class_names[class_id],
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = self._unique_target(target_dir / old_path.name)
            metadata_snapshot = self._metadata_snapshot()
            manifest_existed_before = self._manifest_exists
            delta = _ManifestDelta()
            counts_changed = False

            old_path.replace(target_path)
            try:
                delta = self._update_manifest_change(
                    old_path,
                    target_path,
                    sample.split,
                    class_id,
                )
                self._change_class_counts(
                    sample.split,
                    sample.class_id,
                    class_id,
                )
                counts_changed = True
                self._manifest_exists = True
                self._write_manifest_state()
                self.refresh_metadata()
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "committed",
                        "action": "change_classification",
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "split": sample.split,
                        "old_path": str(old_path),
                        "new_path": str(target_path),
                        "old_class_id": sample.class_id,
                        "new_class_id": class_id,
                        "manifest_existed_before": (
                            manifest_existed_before
                        ),
                        "manifest_changed_rows": [
                            {
                                "before": before,
                                "after": dict(row),
                            }
                            for row, before in delta.changed
                        ],
                        "manifest_appended_rows": [
                            dict(row) for row in delta.appended
                        ],
                    }
                )
            except Exception:
                if target_path.exists() and not old_path.exists():
                    target_path.replace(old_path)
                self._rollback_manifest_delta(delta)
                if counts_changed:
                    self._change_class_counts(
                        sample.split,
                        class_id,
                        sample.class_id,
                    )
                self._manifest_exists = manifest_existed_before
                self._restore_manifest_and_metadata(metadata_snapshot)
                self._append_failed_record(
                    operation_id,
                    "change_classification",
                    old_path,
                )
                raise

            return ClassificationSample(
                image_path=target_path,
                split=sample.split,
                class_id=class_id,
                class_name=self.class_names[class_id],
            )

    def delete_sample(self, sample: ClassificationSample) -> Path:
        with self._lock:
            self._validate_sample(sample)
            self._assert_manifest_unchanged()
            operation_id = uuid.uuid4().hex
            archive_dir = (
                self.archive_root
                / "deleted"
                / datetime.now().strftime("%Y-%m-%d")
                / operation_id
            )
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / sample.image_path.name
            metadata_snapshot = self._metadata_snapshot()
            manifest_existed_before = self._manifest_exists
            delta = _ManifestDelta()
            counts_changed = False

            sample.image_path.replace(archive_path)
            try:
                delta = self._update_manifest_delete(sample.image_path)
                self._delete_counts(sample.split, sample.class_id)
                counts_changed = True
                self._write_manifest_state()
                self.refresh_metadata()
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "committed",
                        "action": "delete_classification",
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "split": sample.split,
                        "image_path": str(sample.image_path),
                        "archive_path": str(archive_path),
                        "class_id": sample.class_id,
                        "manifest_existed_before": (
                            manifest_existed_before
                        ),
                        "manifest_removed_rows": [
                            {"index": index, "row": dict(row)}
                            for index, row in delta.removed
                        ],
                    }
                )
            except Exception:
                if archive_path.exists() and not sample.image_path.exists():
                    sample.image_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    archive_path.replace(sample.image_path)
                self._rollback_manifest_delta(delta)
                if counts_changed:
                    self._restore_deleted_count(
                        sample.split,
                        sample.class_id,
                    )
                self._manifest_exists = manifest_existed_before
                self._restore_manifest_and_metadata(metadata_snapshot)
                self._append_failed_record(
                    operation_id,
                    "delete_classification",
                    sample.image_path,
                )
                raise
            return archive_path

    def undo_latest(self) -> Optional[ClassificationUndoResult]:
        with self._lock:
            self._assert_manifest_unchanged()
            record = self.journal.latest_committed(
                {"change_classification", "delete_classification"}
            )
            if not record:
                return None
            action = str(record.get("action", ""))
            if action == "change_classification":
                return self._undo_class_change(record)
            if action == "delete_classification":
                return self._undo_delete(record)
            return None

    def refresh_metadata(self) -> None:
        counts, split_class_counts, split_image_counts = self._read_counts()
        self._write_data_yaml()
        atomic_write_yaml(
            self.root / "canbang.yaml",
            self._balance_payload(
                counts,
                split_class_counts,
                split_image_counts,
            ),
        )
        atomic_write_json(
            self.root / "stats.json",
            self._stats_payload(split_class_counts, split_image_counts),
        )

    def _undo_class_change(self, record: dict) -> ClassificationUndoResult:
        old_path = _resolve_existing_path(Path(str(record["old_path"])))
        new_path = _resolve_existing_path(Path(str(record["new_path"])))
        split = str(record["split"])
        old_class_id = int(record["old_class_id"])
        new_class_id = int(record["new_class_id"])
        self._validate_undo_paths(new_path, old_path)
        self._validate_class_id(old_class_id)
        self._validate_class_id(new_class_id)
        metadata_snapshot = self._metadata_snapshot()
        manifest_exists_before_undo = self._manifest_exists
        delta = _ManifestDelta()
        counts_changed = False

        old_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.replace(old_path)
        try:
            delta = self._restore_changed_manifest_rows(record)
            self._change_class_counts(split, new_class_id, old_class_id)
            counts_changed = True
            self._manifest_exists = bool(
                record.get("manifest_existed_before", True)
            )
            self._write_manifest_state()
            self.refresh_metadata()
            self._append_undo_record(record, old_path)
        except Exception:
            if old_path.exists() and not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.replace(new_path)
            self._rollback_manifest_delta(delta)
            if counts_changed:
                self._change_class_counts(
                    split,
                    old_class_id,
                    new_class_id,
                )
            self._manifest_exists = manifest_exists_before_undo
            self._restore_manifest_and_metadata(metadata_snapshot)
            raise
        return ClassificationUndoResult(
            action="change_classification",
            sample=ClassificationSample(
                image_path=old_path,
                split=split,
                class_id=old_class_id,
                class_name=self.class_names[old_class_id],
            ),
            replaced_path=new_path,
        )

    def _undo_delete(self, record: dict) -> ClassificationUndoResult:
        image_path = _resolve_existing_path(
            Path(str(record["image_path"]))
        )
        archive_path = _resolve_existing_path(
            Path(str(record["archive_path"]))
        )
        split = str(record["split"])
        class_id = int(record["class_id"])
        self._validate_undo_paths(archive_path, image_path)
        self._validate_class_id(class_id)
        metadata_snapshot = self._metadata_snapshot()
        manifest_exists_before_undo = self._manifest_exists
        delta = _ManifestDelta()
        counts_changed = False

        image_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.replace(image_path)
        try:
            delta = self._restore_deleted_manifest_rows(record)
            self._restore_deleted_count(split, class_id)
            counts_changed = True
            self._manifest_exists = bool(
                record.get("manifest_existed_before", False)
            )
            self._write_manifest_state()
            self.refresh_metadata()
            self._append_undo_record(record, image_path)
        except Exception:
            if image_path.exists() and not archive_path.exists():
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.replace(archive_path)
            self._rollback_manifest_delta(delta)
            if counts_changed:
                self._delete_counts(split, class_id)
            self._manifest_exists = manifest_exists_before_undo
            self._restore_manifest_and_metadata(metadata_snapshot)
            raise
        return ClassificationUndoResult(
            action="delete_classification",
            sample=ClassificationSample(
                image_path=image_path,
                split=split,
                class_id=class_id,
                class_name=self.class_names[class_id],
            ),
            replaced_path=archive_path,
        )

    def _append_undo_record(self, record: dict, image_path: Path) -> None:
        self.journal.append(
            {
                "operation_id": uuid.uuid4().hex,
                "status": "committed",
                "action": "undo",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "target_operation_id": record.get("operation_id"),
                "target_action": record.get("action"),
                "image": str(image_path),
            }
        )

    def _append_failed_record(
        self,
        operation_id: str,
        action: str,
        image_path: Path,
    ) -> None:
        try:
            self.journal.append(
                {
                    "operation_id": operation_id,
                    "status": "failed",
                    "action": action,
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "image": str(image_path),
                }
            )
        except OSError:
            pass

    def _validate_sample(self, sample: ClassificationSample) -> None:
        if not sample.image_path.exists():
            raise ClassificationEditorError(
                "Anh class_f khong con ton tai: {}".format(sample.image_path)
            )
        if sample.split not in SPLITS:
            raise ClassificationEditorError(
                "Split khong hop le: {}".format(sample.split)
            )
        self._validate_class_id(sample.class_id)

    def _validate_class_id(self, class_id: int) -> None:
        if not 0 <= int(class_id) < len(self.class_names):
            raise ClassificationEditorError(
                "Class {} nam ngoai data.yaml".format(class_id)
            )

    def _validate_undo_paths(self, source: Path, target: Path) -> None:
        if not source.exists():
            raise ClassificationEditorError(
                "Khong the hoan tac: file nguon khong con ton tai {}".format(
                    source
                )
            )
        if target.exists():
            raise ClassificationEditorError(
                "Khong the hoan tac vi file dich da ton tai: {}".format(
                    target
                )
            )

    def _unique_target(self, target: Path) -> Path:
        if not target.exists():
            return target
        stem = target.stem
        suffix = target.suffix
        for index in range(1, 10000):
            candidate = target.with_name(
                "{}__classedit{:03d}{}".format(stem, index, suffix)
            )
            if not candidate.exists():
                return candidate
        raise ClassificationEditorError(
            "Khong tao duoc ten file khong trung cho {}".format(target)
        )

    def _metadata_paths(self) -> Tuple[Path, ...]:
        return (
            self.root / "data.yaml",
            self.root / "canbang.yaml",
            self.root / "stats.json",
        )

    def _metadata_snapshot(self) -> Dict[Path, Optional[bytes]]:
        snapshot: Dict[Path, Optional[bytes]] = {}
        for path in self._metadata_paths():
            snapshot[path] = path.read_bytes() if path.exists() else None
        return snapshot

    def _restore_metadata(
        self,
        snapshot: Dict[Path, Optional[bytes]],
    ) -> None:
        for path, content in snapshot.items():
            if content is None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def _restore_manifest_and_metadata(
        self,
        metadata_snapshot: Dict[Path, Optional[bytes]],
    ) -> None:
        restore_error = None
        try:
            self._write_manifest_state()
        except Exception as error:
            restore_error = error
        try:
            self._restore_metadata(metadata_snapshot)
        except Exception as error:
            restore_error = restore_error or error
        if restore_error is not None:
            raise ClassificationEditorError(
                "Khong khoi phuc duoc metadata sau loi: {}".format(
                    restore_error
                )
            ) from restore_error

    def _read_counts(
        self,
    ) -> Tuple[Counter, Dict[str, Counter], Dict[str, int]]:
        return (
            Counter(self._counts),
            {
                split: Counter(self._split_class_counts[split])
                for split in SPLITS
            },
            dict(self._split_image_counts),
        )

    def _change_class_counts(
        self,
        split: str,
        old_class_id: int,
        new_class_id: int,
    ) -> None:
        if old_class_id == new_class_id:
            return
        if (
            self._counts[old_class_id] <= 0
            or self._split_class_counts[split][old_class_id] <= 0
        ):
            raise ClassificationEditorError(
                "Bo dem class_f khong khop voi file dang sua"
            )
        self._counts[old_class_id] -= 1
        self._counts[new_class_id] += 1
        self._split_class_counts[split][old_class_id] -= 1
        self._split_class_counts[split][new_class_id] += 1

    def _delete_counts(self, split: str, class_id: int) -> None:
        if (
            self._counts[class_id] <= 0
            or self._split_class_counts[split][class_id] <= 0
            or self._split_image_counts[split] <= 0
        ):
            raise ClassificationEditorError(
                "Bo dem class_f khong khop voi file dang xoa"
            )
        self._counts[class_id] -= 1
        self._split_class_counts[split][class_id] -= 1
        self._split_image_counts[split] -= 1

    def _restore_deleted_count(self, split: str, class_id: int) -> None:
        self._counts[class_id] += 1
        self._split_class_counts[split][class_id] += 1
        self._split_image_counts[split] += 1

    def _write_data_yaml(self) -> None:
        path = self.root / "data.yaml"
        payload = load_yaml(path)
        updated = dict(payload)
        updated.update(
            {
                "format": "classification_folder",
                "path": ".",
                "train": "train",
                "val": "val",
                "test": "test",
                "nc": len(self.class_names),
                "class_name_mode": "raw",
                "image_size": [
                    CLASSIFICATION_IMAGE_SIZE,
                    CLASSIFICATION_IMAGE_SIZE,
                ],
                "resize_mode": "letterbox",
                "names": {
                    index: str(name)
                    for index, name in enumerate(self.class_names)
                },
            }
        )
        if updated != payload:
            atomic_write_yaml(path, updated)

    def _balance_payload(
        self,
        counts: Counter,
        split_class_counts: Dict[str, Counter],
        split_image_counts: Dict[str, int],
    ) -> dict:
        total_objects = int(sum(counts.values()))
        classes = {}
        for class_id in range(len(self.class_names)):
            count = int(counts[class_id])
            ratio = count / total_objects if total_objects else 0.0
            classes[str(class_id)] = {
                "count": count,
                "ratio": round(ratio, 6),
                "percent": round(ratio * 100.0, 2),
            }
        return {
            "dataset_balance": {
                "version_note": "Updated from class_f editor by CVAT Nhai",
                "total_images": total_objects,
                "total_objects": total_objects,
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
                            for class_id in range(len(self.class_names))
                        },
                    }
                    for split in SPLITS
                },
            }
        }

    def _stats_payload(
        self,
        split_class_counts: Dict[str, Counter],
        split_image_counts: Dict[str, int],
    ) -> dict:
        return {
            "mode": "crop-box",
            "output_size": [
                CLASSIFICATION_IMAGE_SIZE,
                CLASSIFICATION_IMAGE_SIZE,
            ],
            "resize_mode": "letterbox",
            "splits": {
                split: {
                    "images": int(split_image_counts[split]),
                    "classes": {
                        safe_stem(self.class_names[class_id]): int(
                            split_class_counts[split][class_id]
                        )
                        for class_id in range(len(self.class_names))
                    },
                    "skipped": {},
                }
                for split in SPLITS
            },
        }

    def _manifest_path(self) -> Path:
        return self.root / "manifest.csv"

    def _read_manifest(self) -> Tuple[List[str], List[dict]]:
        path = self._manifest_path()
        if not path.exists():
            return list(DEFAULT_MANIFEST_FIELDS), []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or DEFAULT_MANIFEST_FIELDS)
            rows = [dict(row) for row in reader]
        for field_name in DEFAULT_MANIFEST_FIELDS:
            if field_name not in fieldnames:
                fieldnames.append(field_name)
        return fieldnames, rows

    def _write_manifest(
        self,
        fieldnames: Sequence[str],
        rows: Sequence[dict],
    ) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
        )
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            field_name: row.get(field_name, "")
                            for field_name in fieldnames
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _write_manifest_state(self) -> None:
        path = self._manifest_path()
        if self._manifest_exists:
            self._write_manifest(
                self._manifest_fieldnames,
                self._manifest_rows,
            )
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._manifest_signature = self._file_signature(path)

    def _file_signature(self, path: Path) -> Optional[Tuple[int, int]]:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_size, stat.st_mtime_ns

    def _assert_manifest_unchanged(self) -> None:
        if self._file_signature(self._manifest_path()) != (
            self._manifest_signature
        ):
            raise ClassificationEditorError(
                "manifest.csv da bi thay doi ben ngoai; hay mo lai class_f "
                "de tranh ghi de du lieu"
            )

    def _manifest_path_key(self, path: Path) -> str:
        return os.path.normcase(
            os.path.abspath(os.fspath(path.expanduser()))
        ).casefold()

    def _manifest_value_key(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        raw = Path(str(value))
        candidate = raw if raw.is_absolute() else self.root / raw
        return self._manifest_path_key(candidate)

    def _manifest_keys_for_row(self, row: dict) -> Tuple[str, ...]:
        keys = []
        for field_name in MANIFEST_PATH_FIELDS:
            value = row.get(field_name)
            key = self._manifest_value_key(value)
            if key is not None and key not in keys:
                keys.append(key)
            for rebased_key in self._rebased_manifest_keys(row, value):
                if rebased_key not in keys:
                    keys.append(rebased_key)
        return tuple(keys)

    def _rebased_manifest_keys(
        self,
        row: dict,
        value: Optional[str],
    ) -> Tuple[str, ...]:
        """Index absolute manifest paths after a dataset is copied/moved."""
        if not value:
            return ()
        split = str(row.get("split", "")).strip().lower()
        if split not in SPLITS:
            return ()
        filename = Path(str(value)).name
        if not filename:
            return ()

        class_names = []
        try:
            class_id = int(str(row.get("class_id", "")))
        except (TypeError, ValueError):
            class_id = -1
        if 0 <= class_id < len(self.class_names):
            class_names.append(self.class_names[class_id])
        manifest_class = str(row.get("class_name", "")).strip()
        if manifest_class and manifest_class not in class_names:
            class_names.append(manifest_class)

        keys = []
        raw_parts = Path(str(value)).parts
        for class_name in class_names:
            for folder in _candidate_class_folders(
                self.root,
                split,
                class_name,
            ):
                relative_parts: Tuple[str, ...] = ()
                for index in range(len(raw_parts) - 1):
                    if (
                        raw_parts[index].casefold() == split.casefold()
                        and raw_parts[index + 1].casefold()
                        == folder.name.casefold()
                    ):
                        relative_parts = tuple(raw_parts[index + 2 :])
                if not relative_parts:
                    relative_parts = (filename,)
                if any(
                    part in {"", ".", ".."} for part in relative_parts
                ):
                    continue
                key = self._manifest_path_key(
                    folder.joinpath(*relative_parts)
                )
                if key not in keys:
                    keys.append(key)
        return tuple(keys)

    def _rebuild_manifest_index(self) -> None:
        self._manifest_rows_by_path.clear()
        for row in self._manifest_rows:
            self._index_manifest_row(row)

    def _index_manifest_row(self, row: dict) -> None:
        for key in self._manifest_keys_for_row(row):
            bucket = self._manifest_rows_by_path.setdefault(key, [])
            if not any(value is row for value in bucket):
                bucket.append(row)

    def _deindex_manifest_row(self, row: dict) -> None:
        for key in self._manifest_keys_for_row(row):
            bucket = self._manifest_rows_by_path.get(key, [])
            bucket[:] = [value for value in bucket if value is not row]
            if not bucket:
                self._manifest_rows_by_path.pop(key, None)

    def _manifest_rows_for_path(self, path: Path) -> List[dict]:
        return list(
            self._manifest_rows_by_path.get(
                self._manifest_path_key(path),
                (),
            )
        )

    def _format_manifest_path(self, previous: str, path: Path) -> str:
        if previous:
            previous_path = Path(previous)
            if not previous_path.is_absolute():
                try:
                    return str(path.relative_to(self.root))
                except ValueError:
                    return str(path)
        return str(path)

    def _update_manifest_change(
        self,
        old_path: Path,
        new_path: Path,
        split: str,
        class_id: int,
    ) -> _ManifestDelta:
        delta = _ManifestDelta()
        rows = self._manifest_rows_for_path(old_path)
        class_name = safe_stem(self.class_names[class_id])
        for row in rows:
            before = dict(row)
            self._deindex_manifest_row(row)
            previous_output = row.get("output_image", "")
            row["split"] = split
            row["output_image"] = self._format_manifest_path(
                previous_output,
                new_path,
            )
            row["class_id"] = str(class_id)
            row["class_name"] = class_name
            self._index_manifest_row(row)
            delta.changed.append((row, before))
        if not rows:
            row = {
                "split": split,
                "source_image": "",
                "output_image": str(new_path),
                "class_id": str(class_id),
                "class_name": class_name,
                "source": "",
                "source_split": split,
                "leakage_group": "",
            }
            self._manifest_rows.append(row)
            self._index_manifest_row(row)
            delta.appended.append(row)
        return delta

    def _update_manifest_delete(self, path: Path) -> _ManifestDelta:
        delta = _ManifestDelta()
        if not self._manifest_exists:
            return delta
        matched = self._manifest_rows_for_path(path)
        matched_ids = {id(row) for row in matched}
        for index, row in enumerate(self._manifest_rows):
            if id(row) in matched_ids:
                delta.removed.append((index, row))
                self._deindex_manifest_row(row)
        self._manifest_rows = [
            row for row in self._manifest_rows if id(row) not in matched_ids
        ]
        return delta

    def _rollback_manifest_delta(self, delta: _ManifestDelta) -> None:
        appended_ids = {id(row) for row in delta.appended}
        for row in delta.appended:
            self._deindex_manifest_row(row)
        if appended_ids:
            self._manifest_rows = [
                row
                for row in self._manifest_rows
                if id(row) not in appended_ids
            ]
        for row, before in reversed(delta.changed):
            self._deindex_manifest_row(row)
            row.clear()
            row.update(before)
            self._index_manifest_row(row)
        for index, row in sorted(delta.removed, key=lambda item: item[0]):
            self._manifest_rows.insert(
                min(max(0, index), len(self._manifest_rows)),
                row,
            )
            self._index_manifest_row(row)

    def _find_exact_manifest_row(self, expected: dict) -> Optional[dict]:
        key = self._manifest_value_key(expected.get("output_image"))
        candidates = (
            self._manifest_rows_by_path.get(key, [])
            if key is not None
            else self._manifest_rows
        )
        for row in candidates:
            if row == expected:
                return row
        return None

    def _restore_changed_manifest_rows(self, record: dict) -> _ManifestDelta:
        delta = _ManifestDelta()
        for item in record.get("manifest_changed_rows", []):
            before = dict(item.get("before") or {})
            after = dict(item.get("after") or {})
            row = self._find_exact_manifest_row(after)
            if row is None:
                raise ClassificationEditorError(
                    "Khong the hoan tac: dong manifest da thay doi"
                )
            current = dict(row)
            self._deindex_manifest_row(row)
            row.clear()
            row.update(before)
            self._index_manifest_row(row)
            delta.changed.append((row, current))
        for raw_row in record.get("manifest_appended_rows", []):
            expected = dict(raw_row or {})
            row = self._find_exact_manifest_row(expected)
            if row is None:
                raise ClassificationEditorError(
                    "Khong the hoan tac: dong manifest moi khong con ton tai"
                )
            index = next(
                position
                for position, value in enumerate(self._manifest_rows)
                if value is row
            )
            self._deindex_manifest_row(row)
            self._manifest_rows.pop(index)
            delta.removed.append((index, row))
        return delta

    def _restore_deleted_manifest_rows(self, record: dict) -> _ManifestDelta:
        delta = _ManifestDelta()
        entries = sorted(
            record.get("manifest_removed_rows", []),
            key=lambda item: int(item.get("index", 0)),
        )
        for entry in entries:
            row = dict(entry.get("row") or {})
            if self._find_exact_manifest_row(row) is not None:
                raise ClassificationEditorError(
                    "Khong the hoan tac: dong manifest da ton tai"
                )
            index = min(
                max(0, int(entry.get("index", len(self._manifest_rows)))),
                len(self._manifest_rows),
            )
            self._manifest_rows.insert(index, row)
            self._index_manifest_row(row)
            delta.appended.append(row)
        return delta
