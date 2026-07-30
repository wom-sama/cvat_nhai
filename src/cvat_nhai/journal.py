import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class OperationJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def records(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        with self.path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return result

    def latest_committed(
        self,
        actions: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        allowed_actions = set(actions or {"annotate", "delete"})
        undone = set()
        for record in reversed(self.records()):
            if (
                record.get("action") == "undo"
                and record.get("status") == "committed"
            ):
                undone.add(record.get("target_operation_id"))
            elif (
                record.get("status") == "committed"
                and record.get("operation_id") not in undone
                and record.get("action") in allowed_actions
            ):
                return record
        return {}
