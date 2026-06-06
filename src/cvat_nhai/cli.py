import argparse
import json
from pathlib import Path

from .constants import DEFAULT_CLASSIFICATION_ROOT, DEFAULT_DETECTION_ROOT
from .migration import apply_migration, plan_migration
from .schema import audit_schema


def main() -> int:
    parser = argparse.ArgumentParser(prog="cvat-nhai-cli")
    parser.add_argument(
        "--detection-root",
        type=Path,
        default=DEFAULT_DETECTION_ROOT,
    )
    parser.add_argument(
        "--classification-root",
        type=Path,
        default=DEFAULT_CLASSIFICATION_ROOT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="Check five-class YAML schema")
    migrate = subparsers.add_parser(
        "migrate",
        help="Plan or apply four-to-five-class detection migration",
    )
    migrate.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.command == "audit":
        report = audit_schema(
            args.detection_root,
            args.classification_root,
        )
        print(
            json.dumps(
                {
                    "ready": report.ready,
                    "detection_names": report.detection_names,
                    "classification_names": report.classification_names,
                    "messages": report.messages,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if report.ready else 2

    if args.apply:
        report = apply_migration(
            args.detection_root,
            args.classification_root,
        )
    else:
        report, _ = plan_migration(
            args.detection_root,
            args.classification_root / "manifest.csv",
        )
    print(
        json.dumps(
            {
                "total_labels": report.total_labels,
                "total_objects": report.total_objects,
                "changed_labels": report.changed_labels,
                "unresolved_labels": report.unresolved_labels,
                "mismatched_objects": report.mismatched_objects,
                "new_class_counts": report.new_class_counts,
                "issues": report.issues,
                "backup_path": (
                    str(report.backup_path) if report.backup_path else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if not report.unresolved_labels else 3


if __name__ == "__main__":
    raise SystemExit(main())
