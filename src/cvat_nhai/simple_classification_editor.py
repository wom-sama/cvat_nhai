import hashlib
import os
import shutil
import stat
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            self.archive_root / "operations.jsonl"
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
