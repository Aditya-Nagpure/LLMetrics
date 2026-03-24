import json
import threading
from pathlib import Path

from llmetrics.core.exceptions import StorageError
from llmetrics.core.models import TraceRecord
from llmetrics.storage.base import StorageBackend


class JsonlStore(StorageBackend):
    """Append-only, thread-safe JSONL storage. One record per line."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: TraceRecord) -> None:
        try:
            line = record.model_dump_json() + "\n"
            with self._lock:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as e:
            raise StorageError(f"Failed to write trace {record.trace_id}: {e}") from e

    def read_all(self) -> list[TraceRecord]:
        if not self._path.exists():
            return []
        try:
            # Use an ordered dict so later writes (e.g. LLM judge updates) overwrite
            # earlier entries for the same trace_id, preserving insertion order.
            seen: dict[str, TraceRecord] = {}
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = TraceRecord.model_validate_json(line)
                        seen[record.trace_id] = record
            return list(seen.values())
        except (OSError, json.JSONDecodeError) as e:
            raise StorageError(f"Failed to read traces: {e}") from e

    def read_by_id(self, trace_id: str) -> TraceRecord | None:
        # Scan all lines and return the last match (latest update wins).
        result = None
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = TraceRecord.model_validate_json(line)
                        if record.trace_id == trace_id:
                            result = record
        except (OSError, json.JSONDecodeError) as e:
            raise StorageError(f"Failed to read trace {trace_id}: {e}") from e
        return result

    def close(self) -> None:
        pass  # No persistent handle to close