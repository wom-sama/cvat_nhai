import csv
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

from .constants import CLASS_NAMES, IMAGE_EXTENSIONS, SPLITS
from .models import MigrationReport
from .schema import balance_payload, write_five_class_configs
from .utils import atomic_write_text, atomic_write_yaml


def _manifest_index(
    manifest_path: Path,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    exact: Dict[str, List[dict]] = {}
    keys_by_name: DefaultDict[str, set] = defaultdict(set)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            source = Path(row["source_image"])
            key = os.path.normcase(str(source.resolve()))
            exact.setdefault(key, []).append(row)
            keys_by_name[source.name.casefold()].add(key)
    for rows in exact.values():
        rows.sort(key=lambda row: int(row.get("source", 0)))
    fallback = {
        name: exact[next(iter(keys))]
        for name, keys in keys_by_name.items()
        if len(keys) == 1
    }
    return exact, fallback


def _image_for_label(detection_root: Path, split: str, stem: str) -> Optional[Path]:
    folder = detection_root / "images" / split
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / (stem + extension)
        if candidate.exists():
            return candidate
    return None


def plan_migration(
    detection_root: Path,
    manifest_path: Path,
    issue_limit: int = 100,
) -> Tuple[MigrationReport, Dict[Path, str]]:
    exact, fallback = _manifest_index(manifest_path)
    changes = {}
    issues = []
    counts = Counter()
    total_labels = 0
    total_objects = 0
    changed_labels = 0
    unresolved_labels = 0
    mismatched_objects = 0

    for split in SPLITS:
        label_dir = detection_root / "labels" / split
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            total_labels += 1
            image_path = _image_for_label(detection_root, split, label_path.stem)
            if image_path is None:
                unresolved_labels += 1
                if len(issues) < issue_limit:
                    issues.append("Missing image for {}".format(label_path))
                continue

            key = os.path.normcase(str(image_path.resolve()))
            rows = exact.get(key)
            if rows is None:
                rows = fallback.get(image_path.name.casefold())
            old_lines = [
                line
                for line in label_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]
            total_objects += len(old_lines)
            if rows is None or len(rows) != len(old_lines):
                unresolved_labels += 1
                if len(issues) < issue_limit:
                    issues.append(
                        "Manifest mismatch for {}: {} labels, {} manifest rows".format(
                            image_path,
                            len(old_lines),
                            0 if rows is None else len(rows),
                        )
                    )
                continue

            new_lines = []
            label_counts = Counter()
            label_changed = False
            valid = True
            for index, line in enumerate(old_lines):
                parts = line.split()
                if len(parts) < 5:
                    valid = False
                    break
                old_id = int(float(parts[0]))
                manifest_id = int(rows[index]["class_id"])
                if old_id == 0:
                    if manifest_id not in (0, 1):
                        mismatched_objects += 1
                        valid = False
                        if len(issues) < issue_limit:
                            issues.append(
                                "Ambiguous merged class {} row {} has invalid manifest class {}".format(
                                    label_path,
                                    index,
                                    manifest_id,
                                )
                            )
                        break
                    new_id = manifest_id
                elif old_id in (1, 2, 3):
                    new_id = old_id + 1
                    if manifest_id != new_id and len(issues) < issue_limit:
                        issues.append(
                            "Preserved newer detection label {} row {}: old={}, manifest={}".format(
                                label_path,
                                index,
                                old_id,
                                manifest_id,
                            )
                        )
                else:
                    mismatched_objects += 1
                    valid = False
                    if len(issues) < issue_limit:
                        issues.append(
                            "Unsupported old class {} row {}: old={}, manifest={}".format(
                                label_path,
                                index,
                                old_id,
                                manifest_id,
                            )
                        )
                    break
                label_counts[new_id] += 1
                if new_id != old_id:
                    label_changed = True
                new_lines.append(
                    "{} {}".format(new_id, " ".join(parts[1:]))
                )

            if not valid:
                unresolved_labels += 1
                continue
            counts.update(label_counts)
            if label_changed:
                changed_labels += 1
                changes[label_path] = "\n".join(new_lines) + "\n"

    report = MigrationReport(
        total_labels=total_labels,
        total_objects=total_objects,
        changed_labels=changed_labels,
        unresolved_labels=unresolved_labels,
        mismatched_objects=mismatched_objects,
        new_class_counts=dict(counts),
        issues=tuple(issues),
    )
    return report, changes


def apply_migration(
    detection_root: Path,
    classification_root: Path,
    manifest_path: Optional[Path] = None,
) -> MigrationReport:
    manifest_path = manifest_path or classification_root / "manifest.csv"
    report, changes = plan_migration(detection_root, manifest_path)
    if report.unresolved_labels or report.mismatched_objects:
        raise RuntimeError(
            "Migration blocked: {} unresolved labels, {} mismatched objects".format(
                report.unresolved_labels,
                report.mismatched_objects,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = detection_root / "_cvat_nhai_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / ("before_5class_{}.zip".format(timestamp))
    with zipfile.ZipFile(
        str(backup_path),
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(changes):
            archive.write(
                str(path),
                str(path.relative_to(detection_root)),
            )
        for name in ("data.yaml", "canbang.yaml"):
            path = detection_root / name
            if path.exists():
                archive.write(str(path), name)

    for path, content in changes.items():
        atomic_write_text(path, content)

    write_five_class_configs(detection_root, classification_root)
    image_count = 0
    for split in SPLITS:
        image_dir = detection_root / "images" / split
        if image_dir.exists():
            image_count += sum(
                1
                for item in image_dir.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
    atomic_write_yaml(
        detection_root / "canbang.yaml",
        balance_payload(
            report.new_class_counts,
            image_count,
            "Migrated from merged four-class schema using cls_crops manifest",
        ),
    )
    return MigrationReport(
        total_labels=report.total_labels,
        total_objects=report.total_objects,
        changed_labels=report.changed_labels,
        unresolved_labels=report.unresolved_labels,
        mismatched_objects=report.mismatched_objects,
        new_class_counts=report.new_class_counts,
        issues=report.issues,
        backup_path=backup_path,
    )
