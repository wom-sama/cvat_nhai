import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .constants import CLASS_NAMES, SPLITS
from .image_ops import CLASSIFICATION_IMAGE_SIZE
from .models import ClassificationDatasetIndex, ClassificationSample
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
        self._validate_sample(sample)
        self._validate_class_id(class_id)
        if sample.class_id == class_id:
            self.refresh_metadata()
            return sample

        old_path = sample.image_path
        metadata_snapshot = self._metadata_snapshot()
        target_dir = _class_folder(
            self.root,
            sample.split,
            self.class_names[class_id],
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = self._unique_target(target_dir / old_path.name)
        old_path.replace(target_path)
        try:
            self._update_manifest_change(
                old_path,
                target_path,
                sample.split,
                class_id,
            )
            self.refresh_metadata()
        except Exception:
            if target_path.exists() and not old_path.exists():
                target_path.replace(old_path)
            self._restore_metadata(metadata_snapshot)
            raise
        return ClassificationSample(
            image_path=target_path,
            split=sample.split,
            class_id=class_id,
            class_name=self.class_names[class_id],
        )

    def delete_sample(self, sample: ClassificationSample) -> Path:
        self._validate_sample(sample)
        metadata_snapshot = self._metadata_snapshot()
        archive_dir = (
            self.root
            / ".cvat_nhai_classification_archive"
            / "deleted"
            / sample.split
            / safe_stem(sample.class_name)
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._unique_target(archive_dir / sample.image_path.name)
        sample.image_path.replace(archive_path)
        try:
            self._update_manifest_delete(sample.image_path)
            self.refresh_metadata()
        except Exception:
            if archive_path.exists() and not sample.image_path.exists():
                sample.image_path.parent.mkdir(parents=True, exist_ok=True)
                archive_path.replace(sample.image_path)
            self._restore_metadata(metadata_snapshot)
            raise
        return archive_path

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
            self.root / "manifest.csv",
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

    def _read_counts(
        self,
    ) -> Tuple[Counter, Dict[str, Counter], Dict[str, int]]:
        counts = Counter()
        split_class_counts = {split: Counter() for split in SPLITS}
        split_image_counts = {split: 0 for split in SPLITS}
        for split in SPLITS:
            seen = set()
            for class_id, class_name in enumerate(self.class_names):
                for folder in _candidate_class_folders(
                    self.root,
                    split,
                    class_name,
                ):
                    if not folder.is_dir():
                        continue
                    for image_path in scan_images(folder):
                        resolved = _resolve_existing_path(image_path)
                        key = str(resolved).casefold()
                        if key in seen:
                            continue
                        seen.add(key)
                        counts[class_id] += 1
                        split_class_counts[split][class_id] += 1
                        split_image_counts[split] += 1
        return counts, split_class_counts, split_image_counts

    def _write_data_yaml(self) -> None:
        path = self.root / "data.yaml"
        payload = load_yaml(path)
        payload.update(
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
        atomic_write_yaml(path, payload)

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
        for field in DEFAULT_MANIFEST_FIELDS:
            if field not in fieldnames:
                fieldnames.append(field)
        return fieldnames, rows

    def _write_manifest(
        self,
        fieldnames: Sequence[str],
        rows: Sequence[dict],
    ) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".{}.tmp".format(path.name))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: row.get(field, "") for field in fieldnames}
                )
        temporary.replace(path)

    def _manifest_value_matches(self, value: Optional[str], path: Path) -> bool:
        if not value:
            return False
        target = _resolve_existing_path(path)
        raw = Path(str(value))
        candidate = raw if raw.is_absolute() else self.root / raw
        return _resolve_existing_path(candidate) == target

    def _row_matches_path(self, row: dict, path: Path) -> bool:
        for field in ("output_image", "classification_crop", "crop_path", "path"):
            if self._manifest_value_matches(row.get(field), path):
                return True
        return False

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
    ) -> None:
        fieldnames, rows = self._read_manifest()
        matched = False
        class_name = safe_stem(self.class_names[class_id])
        for row in rows:
            if not self._row_matches_path(row, old_path):
                continue
            previous_output = row.get("output_image", "")
            row["split"] = split
            row["output_image"] = self._format_manifest_path(
                previous_output,
                new_path,
            )
            row["class_id"] = str(class_id)
            row["class_name"] = class_name
            matched = True
        if not matched:
            rows.append(
                {
                    "split": split,
                    "source_image": "",
                    "output_image": str(new_path),
                    "class_id": str(class_id),
                    "class_name": class_name,
                    "source": "",
                    "source_split": split,
                    "leakage_group": "",
                }
            )
        self._write_manifest(fieldnames, rows)

    def _update_manifest_delete(self, path: Path) -> None:
        manifest = self._manifest_path()
        if not manifest.exists():
            return
        fieldnames, rows = self._read_manifest()
        kept = [row for row in rows if not self._row_matches_path(row, path)]
        self._write_manifest(fieldnames, kept)
