import hashlib
import os
import shutil
import stat
import tempfile
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from .constants import IMAGE_EXTENSIONS, SPLITS
from .journal import OperationJournal
from .models import (
    ClassificationDatasetIndex,
    ClassificationSample,
    ClassificationUndoResult,
)


class SimpleClassificationError(RuntimeError):
    pass


CHANGE_ACTION = "change_simple_classification"
DELETE_ACTION = "delete_simple_classification"
ARCHIVE_DIR_NAME = ".cvat_nhai_simple_archive"


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path))).casefold()


def _relative_path_key(path: Path) -> str:
    return os.path.normcase(os.fspath(path)).casefold()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_reparse_point(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _validate_simple_root(root: Path) -> None:
    if (root / "data.yaml").is_file() and (
        any((root / split).is_dir() for split in SPLITS)
        or (root / "images").is_dir()
        or (root / "labels").is_dir()
    ):
        raise SimpleClassificationError(
            "Thu muc nay co cau truc YOLO/class_f; hay chon dung che do "
            "dataset, khong dung che do phan loai don gian"
        )
    root_images = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if root_images:
        raise SimpleClassificationError(
            "Co {} anh nam truc tiep trong {}; moi anh phai nam trong mot "
            "thu muc class con".format(len(root_images), root)
        )


def _class_directories(root: Path) -> Tuple[Path, ...]:
    directories = []
    for path in root.iterdir():
        if path.name == ARCHIVE_DIR_NAME or path.name.startswith("."):
            continue
        if _is_reparse_point(path):
            raise SimpleClassificationError(
                "Khong ho tro class la symlink/junction: {}".format(path)
            )
        if not path.is_dir():
            continue
        directories.append(path)
    directories.sort(key=lambda path: (path.name.casefold(), path.name))
    folded = [path.name.casefold() for path in directories]
    if len(folded) != len(set(folded)):
        raise SimpleClassificationError(
            "Ten class bi trung khi khong phan biet hoa/thuong"
        )
    return tuple(directories)


def _scan_class_images(class_dir: Path) -> List[Path]:
    result = []
    for current, directory_names, file_names in os.walk(
        class_dir,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        safe_directories = []
        for name in directory_names:
            child = current_path / name
            if _is_reparse_point(child):
                raise SimpleClassificationError(
                    "Khong ho tro symlink/junction trong class: {}".format(
                        child
                    )
                )
            safe_directories.append(name)
        directory_names[:] = safe_directories
        for name in file_names:
            path = current_path / name
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if _is_reparse_point(path):
                raise SimpleClassificationError(
                    "Khong ho tro anh symlink: {}".format(path)
                )
            # os.walk starts from a resolved absolute class path, so resolving
            # every image again only adds filesystem I/O on large datasets.
            result.append(path)
    result.sort(
        key=lambda path: _relative_path_key(path.relative_to(class_dir))
    )
    return result


def scan_simple_classification_dataset(
    source: Path,
) -> ClassificationDatasetIndex:
    selected = Path(source).expanduser()
    if _is_reparse_point(selected):
        raise SimpleClassificationError(
            "Khong ho tro thu muc data la symlink/junction: {}".format(
                selected
            )
        )
    root = _resolve(selected)
    if not root.is_dir():
        raise SimpleClassificationError(
            "Thu muc data khong ton tai: {}".format(root)
        )
    _validate_simple_root(root)
    class_dirs = _class_directories(root)
    if not class_dirs:
        raise SimpleClassificationError(
            "Khong tim thay thu muc class con trong {}".format(root)
        )

    class_names = tuple(path.name for path in class_dirs)
    samples = []
    for class_id, class_dir in enumerate(class_dirs):
        for image_path in _scan_class_images(class_dir):
            samples.append(
                ClassificationSample(
                    image_path=image_path,
                    split="data",
                    class_id=class_id,
                    class_name=class_dir.name,
                )
            )
    samples.sort(
        key=lambda sample: (
            sample.class_id,
            _relative_path_key(sample.image_path.relative_to(root)),
        )
    )
    return ClassificationDatasetIndex(
        root=root,
        data_yaml=None,
        class_names=class_names,
        samples=tuple(samples),
    )


def _default_archive_root(root: Path) -> Path:
    identity = hashlib.sha256(
        _path_key(root).encode("utf-8")
    ).hexdigest()[:12]
    return root.parent / ARCHIVE_DIR_NAME / identity


class SimpleClassificationEditor:
    def __init__(
        self,
        index: ClassificationDatasetIndex,
        archive_root: Optional[Path] = None,
        create_journal_parent: bool = True,
    ) -> None:
        self.index = index
        self.root = _resolve(index.root)
        self.class_names = tuple(index.class_names)
        self.class_dirs = tuple(
            _resolve(self.root / class_name)
            for class_name in self.class_names
        )
        self.archive_root = _resolve(
            archive_root or _default_archive_root(self.root)
        )
        if _is_relative_to(self.archive_root, self.root):
            raise SimpleClassificationError(
                "Archive cua che do don gian phai nam ngoai thu muc data"
            )
        self.journal = OperationJournal(
            self.archive_root / "operations.jsonl",
            create_parent=create_journal_parent,
        )
        self._lock = threading.RLock()
        self._counts = Counter(
            sample.class_id for sample in self.index.samples
        )

    @property
    def class_counts(self) -> Tuple[int, ...]:
        with self._lock:
            return tuple(
                int(self._counts[class_id])
                for class_id in range(len(self.class_names))
            )

    @property
    def total_images(self) -> int:
        return int(sum(self.class_counts))

    def change_class(
        self,
        sample: ClassificationSample,
        class_id: int,
    ) -> ClassificationSample:
        with self._lock:
            self._validate_sample(sample)
            self._validate_class_id(class_id)
            if class_id == sample.class_id:
                return sample
            self._validate_class_directory(class_id)

            operation_id = uuid.uuid4().hex
            relative = sample.image_path.relative_to(
                self.class_dirs[sample.class_id]
            )
            requested_target = self.class_dirs[class_id] / relative
            self._prepare_class_parent(requested_target.parent, class_id)
            target = self._unique_target(requested_target)
            sample.image_path.replace(target)
            counts_changed = False
            try:
                self._move_count(sample.class_id, class_id)
                counts_changed = True
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "committed",
                        "action": CHANGE_ACTION,
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "old_path": str(sample.image_path),
                        "new_path": str(target),
                        "old_class_id": sample.class_id,
                        "new_class_id": class_id,
                    }
                )
            except Exception:
                if target.exists() and not sample.image_path.exists():
                    sample.image_path.parent.mkdir(parents=True, exist_ok=True)
                    target.replace(sample.image_path)
                if counts_changed:
                    self._move_count(class_id, sample.class_id)
                self._append_failed_record(
                    operation_id,
                    CHANGE_ACTION,
                    sample.image_path,
                )
                raise
            return ClassificationSample(
                image_path=target,
                split="data",
                class_id=class_id,
                class_name=self.class_names[class_id],
            )

    def delete_sample(self, sample: ClassificationSample) -> Path:
        with self._lock:
            self._validate_sample(sample)
            operation_id = uuid.uuid4().hex
            relative = sample.image_path.relative_to(
                self.class_dirs[sample.class_id]
            )
            archive_path = (
                self.archive_root
                / "deleted"
                / datetime.now().strftime("%Y-%m-%d")
                / operation_id
                / sample.class_name
                / relative
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sample.image_path), str(archive_path))
            count_changed = False
            try:
                self._delete_count(sample.class_id)
                count_changed = True
                self.journal.append(
                    {
                        "operation_id": operation_id,
                        "status": "committed",
                        "action": DELETE_ACTION,
                        "timestamp": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "image_path": str(sample.image_path),
                        "archive_path": str(archive_path),
                        "class_id": sample.class_id,
                    }
                )
            except Exception:
                if archive_path.exists() and not sample.image_path.exists():
                    sample.image_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archive_path), str(sample.image_path))
                if count_changed:
                    self._restore_deleted_count(sample.class_id)
                self._append_failed_record(
                    operation_id,
                    DELETE_ACTION,
                    sample.image_path,
                )
                raise
            return archive_path

    def undo_latest(self) -> Optional[ClassificationUndoResult]:
        with self._lock:
            record = self.journal.latest_committed(
                {CHANGE_ACTION, DELETE_ACTION}
            )
            if not record:
                return None
            action = str(record.get("action", ""))
            if action == CHANGE_ACTION:
                return self._undo_change(record)
            if action == DELETE_ACTION:
                return self._undo_delete(record)
            return None

    def _undo_change(self, record: Dict[str, object]) -> ClassificationUndoResult:
        old_path = _resolve(Path(str(record.get("old_path", ""))))
        new_path = _resolve(Path(str(record.get("new_path", ""))))
        old_class_id = int(record.get("old_class_id", -1))
        new_class_id = int(record.get("new_class_id", -1))
        self._validate_class_id(old_class_id)
        self._validate_class_id(new_class_id)
        self._validate_class_directory(old_class_id)
        self._validate_class_directory(new_class_id)
        self._validate_data_path(new_path, new_class_id)
        self._validate_data_path(old_path, old_class_id)
        self._validate_undo_paths(new_path, old_path)

        self._prepare_class_parent(old_path.parent, old_class_id)
        new_path.replace(old_path)
        counts_changed = False
        try:
            self._move_count(new_class_id, old_class_id)
            counts_changed = True
            self._append_undo_record(record, old_path)
        except Exception:
            if old_path.exists() and not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.replace(new_path)
            if counts_changed:
                self._move_count(old_class_id, new_class_id)
            raise
        return ClassificationUndoResult(
            action="change_classification",
            sample=ClassificationSample(
                image_path=old_path,
                split="data",
                class_id=old_class_id,
                class_name=self.class_names[old_class_id],
            ),
            replaced_path=new_path,
        )

    def _undo_delete(self, record: Dict[str, object]) -> ClassificationUndoResult:
        image_path = _resolve(Path(str(record.get("image_path", ""))))
        archive_path = _resolve(Path(str(record.get("archive_path", ""))))
        class_id = int(record.get("class_id", -1))
        self._validate_class_id(class_id)
        self._validate_class_directory(class_id)
        self._validate_data_path(image_path, class_id)
        if not _is_relative_to(archive_path, self.archive_root):
            raise SimpleClassificationError(
                "Duong dan archive nam ngoai archive cua dataset"
            )
        self._validate_undo_paths(archive_path, image_path)

        self._prepare_class_parent(image_path.parent, class_id)
        shutil.move(str(archive_path), str(image_path))
        count_changed = False
        try:
            self._restore_deleted_count(class_id)
            count_changed = True
            self._append_undo_record(record, image_path)
        except Exception:
            if image_path.exists() and not archive_path.exists():
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(image_path), str(archive_path))
            if count_changed:
                self._delete_count(class_id)
            raise
        return ClassificationUndoResult(
            action="delete_classification",
            sample=ClassificationSample(
                image_path=image_path,
                split="data",
                class_id=class_id,
                class_name=self.class_names[class_id],
            ),
            replaced_path=archive_path,
        )

    def _validate_sample(self, sample: ClassificationSample) -> None:
        self._validate_class_id(sample.class_id)
        if sample.class_name != self.class_names[sample.class_id]:
            raise SimpleClassificationError(
                "Class cua sample khong khop voi thu muc data"
            )
        self._validate_data_path(sample.image_path, sample.class_id)
        if sample.image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise SimpleClassificationError(
                "File khong co dinh dang anh duoc ho tro: {}".format(
                    sample.image_path
                )
            )
        if not sample.image_path.is_file():
            raise SimpleClassificationError(
                "Anh khong con ton tai: {}".format(sample.image_path)
            )
        if _is_reparse_point(sample.image_path):
            raise SimpleClassificationError(
                "Khong ho tro anh symlink: {}".format(sample.image_path)
            )

    def _validate_class_id(self, class_id: int) -> None:
        if not 0 <= int(class_id) < len(self.class_names):
            raise SimpleClassificationError(
                "Class {} khong con trong dataset".format(class_id)
            )

    def _validate_class_directory(self, class_id: int) -> None:
        path = self.class_dirs[class_id]
        if not path.is_dir() or _is_reparse_point(path):
            raise SimpleClassificationError(
                "Thu muc class da bi doi/xoa ben ngoai; hay mo lai dataset: "
                "{}".format(path)
            )

    def _prepare_class_parent(self, parent: Path, class_id: int) -> None:
        self._validate_class_directory(class_id)
        class_dir = self.class_dirs[class_id]
        try:
            relative = parent.relative_to(class_dir)
        except ValueError as error:
            raise SimpleClassificationError(
                "Thu muc dich nam ngoai class du kien: {}".format(parent)
            ) from error

        current = class_dir
        for part in relative.parts:
            current = current / part
            if _is_reparse_point(current):
                raise SimpleClassificationError(
                    "Khong ho tro symlink/junction trong class dich: {}".format(
                        current
                    )
                )
            if current.exists():
                if not current.is_dir():
                    raise SimpleClassificationError(
                        "Thanh phan duong dan dich khong phai thu muc: {}".format(
                            current
                        )
                    )
                continue
            current.mkdir()
            if _is_reparse_point(current):
                raise SimpleClassificationError(
                    "Thu muc dich vua tao bi thay bang symlink/junction: {}".format(
                        current
                    )
                )

        if not _is_relative_to(_resolve(current), class_dir):
            raise SimpleClassificationError(
                "Thu muc dich nam ngoai class du kien: {}".format(current)
            )

    def _validate_data_path(self, path: Path, class_id: int) -> None:
        resolved = _resolve(path)
        if not _is_relative_to(resolved, self.class_dirs[class_id]):
            raise SimpleClassificationError(
                "Duong dan anh nam ngoai class du kien: {}".format(path)
            )

    def _validate_undo_paths(self, source: Path, target: Path) -> None:
        if not source.is_file():
            raise SimpleClassificationError(
                "Khong the hoan tac: file nguon khong con ton tai {}".format(
                    source
                )
            )
        if target.exists():
            raise SimpleClassificationError(
                "Khong the hoan tac vi file dich da ton tai: {}".format(
                    target
                )
            )

    def _unique_target(self, target: Path) -> Path:
        if not target.exists():
            return target
        for index in range(1, 10000):
            candidate = target.with_name(
                "{}__classedit{:03d}{}".format(
                    target.stem,
                    index,
                    target.suffix,
                )
            )
            if not candidate.exists():
                return candidate
        raise SimpleClassificationError(
            "Khong tao duoc ten file khong trung cho {}".format(target)
        )

    def _move_count(self, old_class_id: int, new_class_id: int) -> None:
        if self._counts[old_class_id] <= 0:
            raise SimpleClassificationError(
                "Bo dem class khong khop voi file dang di chuyen"
            )
        self._counts[old_class_id] -= 1
        self._counts[new_class_id] += 1

    def _delete_count(self, class_id: int) -> None:
        if self._counts[class_id] <= 0:
            raise SimpleClassificationError(
                "Bo dem class khong khop voi file dang xoa"
            )
        self._counts[class_id] -= 1

    def _restore_deleted_count(self, class_id: int) -> None:
        self._counts[class_id] += 1

    def _append_undo_record(
        self,
        record: Dict[str, object],
        image_path: Path,
    ) -> None:
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
        except Exception:
            pass


ProgressCallback = Optional[Callable[[Dict[str, object]], None]]


@dataclass(frozen=True)
class SimpleClassificationExportReport:
    destination: Path
    images: int
    changed: int
    deleted: int
    class_counts: Tuple[int, ...]
    revision: int


@dataclass(frozen=True)
class _StagedOperation:
    action: str
    before: ClassificationSample
    after: Optional[ClassificationSample]


class StagedSimpleClassificationEditor(SimpleClassificationEditor):
    """Edit class assignments in memory and export an immutable source copy."""

    def __init__(self, index: ClassificationDatasetIndex) -> None:
        super().__init__(index, create_journal_parent=False)
        self._original_samples = {
            _path_key(sample.image_path): sample for sample in index.samples
        }
        if len(self._original_samples) != len(index.samples):
            raise SimpleClassificationError(
                "Dataset co duong dan anh bi trung khi khong phan biet hoa/thuong"
            )
        self._current_samples = dict(self._original_samples)
        self._history: List[_StagedOperation] = []
        self._source_changed_keys: Set[str] = set()
        self._exported_classes: Dict[str, Optional[int]] = {
            key: sample.class_id
            for key, sample in self._original_samples.items()
        }
        self._unexported_keys: Set[str] = set()
        self._revision = 0

    @property
    def staged_change_count(self) -> int:
        with self._lock:
            return len(self._source_changed_keys)

    @property
    def changed_count(self) -> int:
        with self._lock:
            return sum(
                1
                for key in self._source_changed_keys
                if key in self._current_samples
            )

    @property
    def deleted_count(self) -> int:
        with self._lock:
            return sum(
                1
                for key in self._source_changed_keys
                if key not in self._current_samples
            )

    @property
    def has_unexported_changes(self) -> bool:
        with self._lock:
            return bool(self._unexported_keys)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def change_class(
        self,
        sample: ClassificationSample,
        class_id: int,
    ) -> ClassificationSample:
        with self._lock:
            key, current = self._validate_staged_sample(sample)
            self._validate_class_id(class_id)
            self._validate_class_directory(class_id)
            if class_id == current.class_id:
                return current
            changed = ClassificationSample(
                image_path=current.image_path,
                split="data",
                class_id=class_id,
                class_name=self.class_names[class_id],
            )
            self._move_count(current.class_id, class_id)
            self._current_samples[key] = changed
            self._history.append(
                _StagedOperation("change", current, changed)
            )
            self._revision += 1
            self._refresh_change_flags(key)
            return changed

    def delete_sample(self, sample: ClassificationSample) -> Path:
        with self._lock:
            key, current = self._validate_staged_sample(sample)
            self._delete_count(current.class_id)
            self._current_samples.pop(key)
            self._history.append(_StagedOperation("delete", current, None))
            self._revision += 1
            self._refresh_change_flags(key)
            return current.image_path

    def undo_latest(self) -> Optional[ClassificationUndoResult]:
        with self._lock:
            if not self._history:
                return None
            operation = self._history[-1]
            key = _path_key(operation.before.image_path)
            original = self._original_samples[key]
            self._validate_sample(original)
            if operation.action == "change":
                current = self._current_samples.get(key)
                if operation.after is None or current != operation.after:
                    raise SimpleClassificationError(
                        "Trang thai staging da thay doi; khong the hoan tac"
                    )
                self._move_count(
                    operation.after.class_id,
                    operation.before.class_id,
                )
                self._current_samples[key] = operation.before
                result = ClassificationUndoResult(
                    action="change_classification",
                    sample=operation.before,
                    replaced_path=operation.before.image_path,
                )
            elif operation.action == "delete":
                if key in self._current_samples:
                    raise SimpleClassificationError(
                        "Trang thai staging da thay doi; khong the hoan tac"
                    )
                self._restore_deleted_count(operation.before.class_id)
                self._current_samples[key] = operation.before
                result = ClassificationUndoResult(
                    action="delete_classification",
                    sample=operation.before,
                    replaced_path=operation.before.image_path,
                )
            else:
                raise SimpleClassificationError(
                    "Thao tac staging khong hop le"
                )
            self._history.pop()
            self._revision += 1
            self._refresh_change_flags(key)
            return result

    def mark_exported(self, revision: int) -> None:
        with self._lock:
            if revision != self._revision:
                raise SimpleClassificationError(
                    "Phien sua da thay doi trong luc export; hay xuat lai"
                )
            self._exported_classes = {
                key: (
                    self._current_samples[key].class_id
                    if key in self._current_samples
                    else None
                )
                for key in self._original_samples
            }
            self._unexported_keys.clear()

    def export_dataset(
        self,
        destination: Path,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> SimpleClassificationExportReport:
        destination = self._prepare_export_destination(destination)
        destination_existed = destination.exists()
        snapshot, revision, changed, deleted, counts = self._export_snapshot()
        maximum = max(1, len(snapshot) + 1)
        self._report_progress(
            progress_callback,
            "Dang chuan bi dataset moi",
            "{} anh se duoc sao chep; dataset nguon duoc giu nguyen".format(
                len(snapshot)
            ),
            0,
            maximum,
        )
        self._check_cancelled(cancel_event)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".cvat_nhai_simple_export_",
                dir=str(destination.parent),
            )
        )
        removed_empty_destination = False
        try:
            for class_name in self.class_names:
                (staging / class_name).mkdir(parents=True, exist_ok=False)
            targets = self._allocate_export_targets(snapshot)
            for item_index, ((original, _class_id), relative) in enumerate(
                zip(snapshot, targets),
                start=1,
            ):
                self._check_cancelled(cancel_event)
                self._validate_sample(original)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original.image_path, target)
                self._report_progress(
                    progress_callback,
                    "Dang tao dataset phan loai moi",
                    "{} / {} - {}".format(
                        item_index,
                        len(snapshot),
                        original.image_path.name,
                    ),
                    item_index,
                    maximum,
                )

            self._check_cancelled(cancel_event)
            self._validate_destination_before_publish(
                destination,
                destination_existed,
            )
            if destination_existed:
                destination.rmdir()
                removed_empty_destination = True
            os.replace(str(staging), str(destination))
            self._report_progress(
                progress_callback,
                "Da tao xong dataset moi",
                "{} anh, {} doi class, {} loai bo".format(
                    len(snapshot),
                    changed,
                    deleted,
                ),
                maximum,
                maximum,
            )
            return SimpleClassificationExportReport(
                destination=destination,
                images=len(snapshot),
                changed=changed,
                deleted=deleted,
                class_counts=counts,
                revision=revision,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if removed_empty_destination and not destination.exists():
                destination.mkdir(parents=True, exist_ok=True)
            raise

    def _validate_staged_sample(
        self,
        sample: ClassificationSample,
    ) -> Tuple[str, ClassificationSample]:
        key = _path_key(sample.image_path)
        original = self._original_samples.get(key)
        if original is None:
            raise SimpleClassificationError(
                "Anh khong thuoc phien staging hien tai: {}".format(
                    sample.image_path
                )
            )
        current = self._current_samples.get(key)
        if current is None:
            raise SimpleClassificationError(
                "Anh da duoc danh dau loai khoi dataset moi"
            )
        if (
            sample.class_id != current.class_id
            or sample.class_name != current.class_name
        ):
            raise SimpleClassificationError(
                "Class cua anh da thay doi; hay tai lai anh hien tai"
            )
        self._validate_sample(original)
        return key, current

    def _refresh_change_flags(self, key: str) -> None:
        current = self._current_samples.get(key)
        current_class = current.class_id if current is not None else None
        source_class = self._original_samples[key].class_id
        if current_class == source_class:
            self._source_changed_keys.discard(key)
        else:
            self._source_changed_keys.add(key)
        if current_class == self._exported_classes[key]:
            self._unexported_keys.discard(key)
        else:
            self._unexported_keys.add(key)

    def _export_snapshot(
        self,
    ) -> Tuple[
        Tuple[Tuple[ClassificationSample, int], ...],
        int,
        int,
        int,
        Tuple[int, ...],
    ]:
        with self._lock:
            snapshot = []
            changed = 0
            deleted = 0
            for original in self.index.samples:
                key = _path_key(original.image_path)
                current = self._current_samples.get(key)
                if current is None:
                    deleted += 1
                    continue
                snapshot.append((original, current.class_id))
                if current.class_id != original.class_id:
                    changed += 1
            return (
                tuple(snapshot),
                self._revision,
                changed,
                deleted,
                self.class_counts,
            )

    def _prepare_export_destination(self, selected: Path) -> Path:
        raw = Path(
            os.path.abspath(os.fspath(Path(selected).expanduser()))
        )
        self._validate_no_reparse_ancestors(raw)
        destination = _resolve(raw)
        if _is_relative_to(destination, self.root) or _is_relative_to(
            self.root,
            destination,
        ):
            raise SimpleClassificationError(
                "Thu muc dataset moi phai nam ngoai dataset nguon"
            )
        if destination.exists():
            if not destination.is_dir():
                raise SimpleClassificationError(
                    "Duong dan dataset dich khong phai thu muc"
                )
            if any(destination.iterdir()):
                raise SimpleClassificationError(
                    "Thu muc dataset dich phai rong"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._validate_no_reparse_ancestors(destination.parent)
        return destination

    def _validate_destination_before_publish(
        self,
        destination: Path,
        destination_existed: bool,
    ) -> None:
        self._validate_no_reparse_ancestors(destination)
        if destination_existed:
            if not destination.is_dir() or any(destination.iterdir()):
                raise SimpleClassificationError(
                    "Thu muc dataset dich da thay doi trong luc export"
                )
        elif destination.exists():
            raise SimpleClassificationError(
                "Thu muc dataset dich da xuat hien trong luc export"
            )

    def _validate_no_reparse_ancestors(self, path: Path) -> None:
        current = path
        while True:
            if current.exists() and _is_reparse_point(current):
                raise SimpleClassificationError(
                    "Khong ho tro dich export qua symlink/junction: {}".format(
                        current
                    )
                )
            parent = current.parent
            if parent == current:
                return
            current = parent

    def _allocate_export_targets(
        self,
        snapshot: Tuple[Tuple[ClassificationSample, int], ...],
    ) -> Tuple[Path, ...]:
        used_files: Set[str] = set()
        used_directories: Set[str] = {
            _relative_path_key(Path(name)) for name in self.class_names
        }
        targets = []
        for original, class_id in snapshot:
            relative = original.image_path.relative_to(
                self.class_dirs[original.class_id]
            )
            requested = Path(self.class_names[class_id]) / relative
            candidate = requested
            if self._target_parent_conflicts(candidate, used_files):
                candidate = Path(self.class_names[class_id]) / "__".join(
                    relative.parts
                )
            for suffix_index in range(10000):
                if suffix_index:
                    candidate = candidate.with_name(
                        "{}__classedit{:03d}{}".format(
                            requested.stem,
                            suffix_index,
                            requested.suffix,
                        )
                    )
                    if self._target_parent_conflicts(candidate, used_files):
                        candidate = Path(self.class_names[class_id]) / (
                            "{}__classedit{:03d}{}".format(
                                "__".join(relative.with_suffix("").parts),
                                suffix_index,
                                relative.suffix,
                            )
                        )
                key = _relative_path_key(candidate)
                if (
                    key not in used_files
                    and key not in used_directories
                    and not self._target_parent_conflicts(
                        candidate,
                        used_files,
                    )
                ):
                    break
            else:
                raise SimpleClassificationError(
                    "Khong tao duoc ten file export khong trung cho {}".format(
                        original.image_path
                    )
                )
            used_files.add(_relative_path_key(candidate))
            parent = candidate.parent
            while parent != Path("."):
                used_directories.add(_relative_path_key(parent))
                parent = parent.parent
            targets.append(candidate)
        return tuple(targets)

    @staticmethod
    def _target_parent_conflicts(
        path: Path,
        used_files: Set[str],
    ) -> bool:
        parent = path.parent
        while parent != Path("."):
            if _relative_path_key(parent) in used_files:
                return True
            parent = parent.parent
        return False

    @staticmethod
    def _check_cancelled(
        cancel_event: Optional[threading.Event],
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SimpleClassificationError("Export da bi huy an toan")

    @staticmethod
    def _report_progress(
        callback: ProgressCallback,
        stage: str,
        detail: str,
        value: int,
        maximum: int,
    ) -> None:
        if callback is not None:
            callback(
                {
                    "stage": stage,
                    "detail": detail,
                    "value": value,
                    "maximum": maximum,
                }
            )
