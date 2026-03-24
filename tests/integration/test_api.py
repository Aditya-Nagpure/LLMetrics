"""End-to-end API tests for Phase 5.

Uses FastAPI TestClient with mock pipeline components injected via create_app().
No real ChromaDB, model, or LLM calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from llmetrics.api.app import create_app
from llmetrics.core.models import TokenUsage, TraceRecord
from llmetrics.storage.base import StorageBackend
from llmetrics.tracing.span import span as trace_span


# ---------------------------------------------------------------------------
# Span-emitting pipeline fakes (used to verify the span→trace plumbing)
# ---------------------------------------------------------------------------


class _SpanRetriever:
    def retrieve(self, query: str) -> list[str]:
        with trace_span("retriever"):
            return ["doc_one", "doc_two"]


class _SpanReranker:
    def rerank(self, query: str, docs: list[str]) -> list[str]:
        with trace_span("reranker"):
            return ["doc_one"]


class _SpanPromptBuilder:
    def build(self, query: str, docs: list[str]) -> str:
        with trace_span("prompt_builder"):
            return "assembled prompt"


class _SpanLLMClient:
    def complete(self, prompt: str):
        with trace_span("llm"):
            return "LLM answer", TokenUsage(prompt=10, completion=5, total=15), 0.05


# ---------------------------------------------------------------------------
# In-memory storage stub
# ---------------------------------------------------------------------------


class _MemoryStorage(StorageBackend):
    def __init__(self) -> None:
        self._records: list[TraceRecord] = []

    def write(self, record: TraceRecord) -> None:
        self._records.append(record)

    def read_all(self) -> list[TraceRecord]:
        return list(self._records)

    def read_by_id(self, trace_id: str) -> TraceRecord | None:
        return next((r for r in self._records if r.trace_id == trace_id), None)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage() -> _MemoryStorage:
    return _MemoryStorage()


@pytest.fixture()
def mock_retriever() -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = ["doc_one", "doc_two"]
    return m


@pytest.fixture()
def mock_reranker() -> MagicMock:
    m = MagicMock()
    m.rerank.return_value = ["doc_one"]
    return m


@pytest.fixture()
def mock_prompt_builder() -> MagicMock:
    m = MagicMock()
    m.build.return_value = "assembled prompt"
    return m


@pytest.fixture()
def mock_llm_client() -> MagicMock:
    m = MagicMock()
    m.complete.return_value = (
        "LLM answer",
        TokenUsage(prompt=10, completion=5, total=15),
        0.05,
    )
    return m


@pytest.fixture()
def client(
    storage: _MemoryStorage,
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_client: MagicMock,
) -> TestClient:
    app = create_app(
        storage=storage,
        retriever=mock_retriever,
        reranker=mock_reranker,
        prompt_builder=mock_prompt_builder,
        llm_client=mock_llm_client,
    )
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /query tests
# ---------------------------------------------------------------------------


def test_query_returns_200(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "What is RAG?"})
    assert resp.status_code == 200


def test_query_response_contains_answer_and_trace_id(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "What is RAG?"})
    data = resp.json()
    assert data["response"] == "LLM answer"
    assert "trace_id" in data
    assert len(data["trace_id"]) > 0


def test_query_writes_trace_to_storage(
    client: TestClient, storage: _MemoryStorage
) -> None:
    resp = client.post("/query", json={"query": "explain embeddings"})
    trace_id = resp.json()["trace_id"]

    record = storage.read_by_id(trace_id)
    assert record is not None
    assert record.query == "explain embeddings"
    assert record.response == "LLM answer"


def test_query_trace_has_correct_docs(
    client: TestClient, storage: _MemoryStorage
) -> None:
    client.post("/query", json={"query": "q"})
    record = storage.read_all()[0]

    assert record.retrieved_docs == [{"content": "doc_one"}, {"content": "doc_two"}]
    assert record.reranked_docs == [{"content": "doc_one"}]


def test_query_trace_has_token_usage(
    client: TestClient, storage: _MemoryStorage
) -> None:
    client.post("/query", json={"query": "q"})
    record = storage.read_all()[0]

    assert record.tokens is not None
    assert record.tokens.prompt == 10
    assert record.tokens.completion == 5
    assert record.tokens.total == 15


def test_query_trace_has_latency(
    client: TestClient, storage: _MemoryStorage
) -> None:
    client.post("/query", json={"query": "q"})
    record = storage.read_all()[0]

    assert record.latency_total_s >= 0.0
    assert record.latency_llm_s == pytest.approx(0.05)


def test_query_trace_has_spans() -> None:
    """Verify that spans emitted by pipeline components flow into the TraceRecord."""
    mem = _MemoryStorage()
    app = create_app(
        storage=mem,
        retriever=_SpanRetriever(),
        reranker=_SpanReranker(),
        prompt_builder=_SpanPromptBuilder(),
        llm_client=_SpanLLMClient(),
    )
    with TestClient(app) as c:
        c.post("/query", json={"query": "q"})

    record = mem.read_all()[0]
    span_names = [s.name for s in record.spans]
    assert "retriever" in span_names
    assert "reranker" in span_names
    assert "prompt_builder" in span_names
    assert "llm" in span_names


def test_query_calls_pipeline_in_order(
    client: TestClient,
    mock_retriever: MagicMock,
    mock_reranker: MagicMock,
    mock_prompt_builder: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    client.post("/query", json={"query": "order test"})

    mock_retriever.retrieve.assert_called_once_with("order test")
    mock_reranker.rerank.assert_called_once_with("order test", ["doc_one", "doc_two"])
    mock_prompt_builder.build.assert_called_once_with("order test", ["doc_one"])
    mock_llm_client.complete.assert_called_once_with("assembled prompt")


# ---------------------------------------------------------------------------
# GET /metrics tests
# ---------------------------------------------------------------------------


def test_metrics_empty_before_requests(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 0
    assert data["p50_latency_s"] is None
    assert data["p95_latency_s"] is None


def test_metrics_counts_requests(client: TestClient) -> None:
    client.post("/query", json={"query": "q1"})
    client.post("/query", json={"query": "q2"})

    resp = client.get("/metrics")
    assert resp.json()["total_requests"] == 2


def test_metrics_has_latency_after_query(client: TestClient) -> None:
    client.post("/query", json={"query": "q"})

    resp = client.get("/metrics")
    data = resp.json()
    assert data["p50_latency_s"] is not None
    assert data["p50_latency_s"] >= 0.0