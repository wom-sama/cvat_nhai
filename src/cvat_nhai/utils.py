import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

import yaml

from .constants import CLASS_NAMES
from .image_ops import CLASSIFICATION_IMAGE_SIZE


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "image"


def stable_suffix(path: Path) -> str:
    stat = path.stat()
    payload = "{}|{}|{}".format(path.resolve(), stat.st_size, stat.st_mtime_ns)
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:10]


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix="." + path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def atomic_write_yaml(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
    )


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("{} must contain a YAML mapping".format(path))
    return value


def names_from_yaml(payload: dict) -> List[str]:
    names = payload.get("names") or []
    if isinstance(names, dict):
        def key(value: Any) -> int:
            return int(value)

        return [str(names[item]) for item in sorted(names, key=key)]
    return [str(item) for item in names]


def detection_yaml_payload(root: Path) -> dict:
    return {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
        "names": list(CLASS_NAMES),
    }


def classification_yaml_payload() -> dict:
    return {
        "format": "classification_folder",
        "path": ".",
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(CLASS_NAMES),
        "class_name_mode": "raw",
        "canbang_yaml": "canbang.yaml",
        "image_size": [
            CLASSIFICATION_IMAGE_SIZE,
            CLASSIFICATION_IMAGE_SIZE,
        ],
        "resize_mode": "letterbox",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }


def path_is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False
