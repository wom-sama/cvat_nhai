from pathlib import Path

from cvat_nhai.scanner import scan_images


def test_scan_images_recursive_sorted_and_excluded(tmp_path: Path) -> None:
    root = tmp_path / "source"
    nested = root / "b" / "nested"
    nested.mkdir(parents=True)
    (root / "a.JPG").write_bytes(b"x")
    (nested / "c.png").write_bytes(b"x")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")
    archive = root / ".cvat_nhai_archive"
    archive.mkdir()
    (archive / "deleted.jpg").write_bytes(b"x")
    output = root / "output"
    output.mkdir()
    (output / "generated.jpg").write_bytes(b"x")

    result = scan_images(root, (output,))

    assert [item.name for item in result] == ["a.JPG", "c.png"]
