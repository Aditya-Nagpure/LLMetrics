"""Integration tests for storage backends — parametrized over JsonlStore and SqliteStore."""
import threading
from pathlib import Path

import pytest

from llmetrics.core.models import EvalResult, SpanEvent, TokenUsage, TraceRecord
from llmetrics.storage.jsonl_store import JsonlStore
from llmetrics.storage.sqlite_store import SqliteStore
from llmetrics.storage.base import StorageBackend


def _make_record(**kwargs) -> TraceRecord:
    defaults = dict(
        query="What is RAG?",
        retrieved_docs=[{"id": "doc1", "text": "RAG stands for retrieval augmented generation."}],
        reranked_docs=[{"id": "doc1", "score": 0.95}],
        prompt="Answer using context: ...",
        response="RAG stands for retrieval augmented generation.",
        latency_total_s=0.45,
        latency_llm_s=0.30,
        tokens=TokenUsage(prompt=120, completion=80, total=200),
        cost_usd=0.0002,
        spans=[SpanEvent(name="retriever", start_time=0.0, end_time=0.1, duration_s=0.1)],
        evaluation=EvalResult(faithfulness=0.9),
    )
    defaults.update(kwargs)
    return TraceRecord(**defaults)


@pytest.fixture(params=["jsonl", "sqlite"])
def store(request, tmp_path: Path) -> StorageBackend:
    if request.param == "jsonl":
        s = JsonlStore(tmp_path / "traces.jsonl")
    else:
        s = SqliteStore(tmp_path / "traces.db")
    yield s
    s.close()


class TestWriteReadRoundTrip:
    def test_write_and_read_all(self, store: StorageBackend) -> None:
        record = _make_record()
        store.write(record)
        results = store.read_all()
        assert len(results) == 1
        assert results[0].trace_id == record.trace_id
        assert results[0].query == record.query

    def test_multiple_records(self, store: StorageBackend) -> None:
        records = [_make_record(query=f"query {i}") for i in range(5)]
        for r in records:
            store.write(r)
        results = store.read_all()
        assert len(results) == 5

    def test_read_by_id_found(self, store: StorageBackend) -> None:
        record = _make_record()
        store.write(record)
        result = store.read_by_id(record.trace_id)
        assert result is not None
        assert result.trace_id == record.trace_id

    def test_read_by_id_not_found(self, store: StorageBackend) -> None:
        result = store.read_by_id("nonexistent-id")
        assert result is None

    def test_read_all_empty(self, store: StorageBackend) -> None:
        assert store.read_all() == []

    def test_full_record_roundtrip(self, store: StorageBackend) -> None:
        record = _make_record()
        store.write(record)
        result = store.read_by_id(record.trace_id)
        assert result is not None
        assert result.model_dump() == record.model_dump()

    def test_record_with_error(self, store: StorageBackend) -> None:
        record = _make_record(error="LLM timeout")
        store.write(record)
        result = store.read_by_id(record.trace_id)
        assert result is not None
        assert result.error == "LLM timeout"

    def test_record_with_no_tokens(self, store: StorageBackend) -> None:
        record = _make_record(tokens=None)
        store.write(record)
        result = store.read_by_id(record.trace_id)
        assert result is not None
        assert result.tokens is None


class TestConcurrentWrites:
    def test_concurrent_writes(self, store: StorageBackend) -> None:
        n = 20
        records = [_make_record(query=f"concurrent query {i}") for i in range(n)]
        errors: list[Exception] = []

        def write_record(r: TraceRecord) -> None:
            try:
                store.write(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_record, args=(r,)) for r in records]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        results = store.read_all()
        assert len(results) == n

    def test_concurrent_writes_unique_ids(self, store: StorageBackend) -> None:
        n = 20
        records = [_make_record() for _ in range(n)]
        threads = [threading.Thread(target=store.write, args=(r,)) for r in records]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = store.read_all()
        ids = {r.trace_id for r in results}
        assert len(ids) == n  # All unique IDs persisted


class TestContextManager:
    def test_context_manager_jsonl(self, tmp_path: Path) -> None:
        record = _make_record()
        with JsonlStore(tmp_path / "ctx.jsonl") as store:
            store.write(record)
        # After close, a new store should still read the data
        with JsonlStore(tmp_path / "ctx.jsonl") as store2:
            assert len(store2.read_all()) == 1

    def test_context_manager_sqlite(self, tmp_path: Path) -> None:
        record = _make_record()
        with SqliteStore(tmp_path / "ctx.db") as store:
            store.write(record)
        with SqliteStore(tmp_path / "ctx.db") as store2:
            assert len(store2.read_all()) == 1