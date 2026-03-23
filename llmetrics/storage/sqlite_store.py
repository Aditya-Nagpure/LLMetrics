import json
import sqlite3
import threading
from pathlib import Path

from llmetrics.core.exceptions import StorageError
from llmetrics.core.models import EvalResult, TokenUsage, TraceRecord
from llmetrics.storage.base import StorageBackend

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    query           TEXT NOT NULL,
    retrieved_docs  TEXT NOT NULL,
    reranked_docs   TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    response        TEXT NOT NULL,
    latency_total_s REAL NOT NULL,
    latency_llm_s   REAL NOT NULL,
    tokens          TEXT,
    cost_usd        REAL NOT NULL,
    spans           TEXT NOT NULL,
    evaluation      TEXT NOT NULL,
    error           TEXT
)
"""


def _record_to_row(record: TraceRecord) -> tuple:
    return (
        record.trace_id,
        record.timestamp.isoformat(),
        record.query,
        json.dumps(record.retrieved_docs),
        json.dumps(record.reranked_docs),
        record.prompt,
        record.response,
        record.latency_total_s,
        record.latency_llm_s,
        record.tokens.model_dump_json() if record.tokens else None,
        record.cost_usd,
        json.dumps([s.model_dump() for s in record.spans]),
        record.evaluation.model_dump_json(),
        record.error,
    )


def _row_to_record(row: sqlite3.Row) -> TraceRecord:
    return TraceRecord.model_validate(
        {
            "trace_id": row["trace_id"],
            "timestamp": row["timestamp"],
            "query": row["query"],
            "retrieved_docs": json.loads(row["retrieved_docs"]),
            "reranked_docs": json.loads(row["reranked_docs"]),
            "prompt": row["prompt"],
            "response": row["response"],
            "latency_total_s": row["latency_total_s"],
            "latency_llm_s": row["latency_llm_s"],
            "tokens": json.loads(row["tokens"]) if row["tokens"] else None,
            "cost_usd": row["cost_usd"],
            "spans": json.loads(row["spans"]),
            "evaluation": json.loads(row["evaluation"]),
            "error": row["error"],
        }
    )


class SqliteStore(StorageBackend):
    """SQLite storage with flat columns and JSON blobs for nested fields."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def write(self, record: TraceRecord) -> None:
        try:
            row = _record_to_row(record)
            with self._lock:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO traces
                    (trace_id, timestamp, query, retrieved_docs, reranked_docs,
                     prompt, response, latency_total_s, latency_llm_s, tokens,
                     cost_usd, spans, evaluation, error)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    row,
                )
                self._conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to write trace {record.trace_id}: {e}") from e

    def read_all(self) -> list[TraceRecord]:
        try:
            cursor = self._conn.execute("SELECT * FROM traces ORDER BY timestamp")
            return [_row_to_record(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to read traces: {e}") from e

    def read_by_id(self, trace_id: str) -> TraceRecord | None:
        try:
            cursor = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            )
            row = cursor.fetchone()
            return _row_to_record(row) if row else None
        except sqlite3.Error as e:
            raise StorageError(f"Failed to read trace {trace_id}: {e}") from e

    def close(self) -> None:
        self._conn.close()