from pathlib import Path
from typing import Dict

from .constants import CLASS_NAMES, SPLITS
from .models import SchemaAudit
from .utils import (
    atomic_write_yaml,
    classification_yaml_payload,
    detection_yaml_payload,
    load_yaml,
    names_from_yaml,
)


def audit_schema(detection_root: Path, classification_root: Path) -> SchemaAudit:
    detection_names = tuple(
        names_from_yaml(load_yaml(detection_root / "data.yaml"))
    )
    classification_names = tuple(
        names_from_yaml(load_yaml(classification_root / "data.yaml"))
    )
    messages = []

    if detection_names != CLASS_NAMES:
        messages.append(
            "Detection data.yaml is not using the required five-class schema."
        )
    if classification_names != CLASS_NAMES:
        messages.append(
            "Classification data.yaml is not using the required five-class schema."
        )

    return SchemaAudit(
        ready=not messages,
        detection_names=detection_names,
        classification_names=classification_names,
        messages=tuple(messages),
    )


def ensure_directory_layout(detection_root: Path, classification_root: Path) -> None:
    for split in SPLITS:
        (detection_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (detection_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for class_name in CLASS_NAMES:
            (classification_root / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


def initialize_empty_datasets(
    detection_root: Path,
    classification_root: Path,
) -> None:
    ensure_directory_layout(detection_root, classification_root)
    detection_yaml = detection_root / "data.yaml"
    classification_yaml = classification_root / "data.yaml"
    if not detection_yaml.exists():
        atomic_write_yaml(detection_yaml, detection_yaml_payload(detection_root))
    if not classification_yaml.exists():
        atomic_write_yaml(classification_yaml, classification_yaml_payload())

    manifest = classification_root / "manifest.csv"
    if not manifest.exists():
        manifest.write_text(
            "split,source_image,output_image,class_id,class_name,source\n",
            encoding="utf-8",
        )
    zero_counts = {index: 0 for index in range(len(CLASS_NAMES))}
    if not (detection_root / "canbang.yaml").exists():
        atomic_write_yaml(
            detection_root / "canbang.yaml",
            balance_payload(zero_counts, 0, "Initialized by CVAT Nhai"),
        )
    if not (classification_root / "canbang.yaml").exists():
        atomic_write_yaml(
            classification_root / "canbang.yaml",
            balance_payload(zero_counts, 0, "Initialized by CVAT Nhai"),
        )


def write_five_class_configs(
    detection_root: Path,
    classification_root: Path,
) -> None:
    atomic_write_yaml(
        detection_root / "data.yaml",
        detection_yaml_payload(detection_root),
    )
    atomic_write_yaml(
        classification_root / "data.yaml",
        classification_yaml_payload(),
    )


def balance_payload(
    counts: Dict[int, int],
    total_images: int,
    version_note: str,
) -> dict:
    total_objects = sum(counts.values())
    classes = {}
    for class_id in range(len(CLASS_NAMES)):
        count = int(counts.get(class_id, 0))
        ratio = (count / total_objects) if total_objects else 0.0
        classes[str(class_id)] = {
            "count": count,
            "ratio": round(ratio, 6),
            "percent": round(ratio * 100.0, 2),
        }
    return {
        "dataset_balance": {
            "version_note": version_note,
            "total_images": int(total_images),
            "total_objects": int(total_objects),
            "train_ratio_config": 0.7,
            "val_ratio_config": 0.2,
            "test_ratio_config": 0.1,
            "classes": classes,
        }
    }
