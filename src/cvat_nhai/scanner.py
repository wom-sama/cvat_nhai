import os
from pathlib import Path
from typing import Iterable, List

from .constants import IMAGE_EXTENSIONS
from .utils import path_is_within


SKIPPED_DIR_NAMES = {
    ".git",
    ".cvat_nhai_archive",
    ".cvat_nhai_tmp",
    "__pycache__",
}


def scan_images(root: Path, excluded_roots: Iterable[Path] = ()) -> List[Path]:
    root = root.resolve()
    excluded = tuple(path.resolve() for path in excluded_roots)
    if not root.is_dir():
        return []

    images = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(str(current)))
        except (OSError, PermissionError):
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if entry.name in SKIPPED_DIR_NAMES or entry.name.startswith(".cvat_nhai"):
                    continue
                if path_is_within(path, excluded):
                    continue
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(path)

    return sorted(images, key=lambda value: str(value).casefold())
