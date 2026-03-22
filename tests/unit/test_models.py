import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llmetrics.core.models import EvalResult, SpanEvent, TokenUsage, TraceRecord


class TestTokenUsage:
    def test_round_trip(self):
        obj = TokenUsage(prompt=100, completion=50, total=150)
        assert TokenUsage.model_validate_json(obj.model_dump_json()) == obj

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TokenUsage(prompt=1, completion=1, total=2, unexpected="x")  # type: ignore[call-arg]


class TestSpanEvent:
    def test_round_trip(self):
        obj = SpanEvent(
            name="retriever",
            start_time=1000.0,
            end_time=1000.5,
            duration_s=0.5,
            metadata={"top_k": 10, "collection": "docs"},
        )
        assert SpanEvent.model_validate_json(obj.model_dump_json()) == obj

    def test_default_metadata(self):
        obj = SpanEvent(name="llm", start_time=0.0, end_time=1.0, duration_s=1.0)
        assert obj.metadata == {}

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            SpanEvent(name="x", start_time=0.0, end_time=1.0, duration_s=1.0, bad="y")  # type: ignore[call-arg]


class TestEvalResult:
    def test_all_none_by_default(self):
        obj = EvalResult()
        assert obj.faithfulness is None
        assert obj.context_relevance is None
        assert obj.answer_correctness is None

    def test_round_trip_with_values(self):
        obj = EvalResult(faithfulness=0.9, context_relevance=0.75)
        assert EvalResult.model_validate_json(obj.model_dump_json()) == obj

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvalResult(hallucination_score=0.1)  # type: ignore[call-arg]


class TestTraceRecord:
    def test_defaults(self):
        record = TraceRecord(query="What is RAG?")
        assert record.query == "What is RAG?"
        assert record.trace_id != ""
        assert isinstance(record.timestamp, datetime)
        assert record.retrieved_docs == []
        assert record.reranked_docs == []
        assert record.spans == []
        assert record.tokens is None
        assert record.error is None

    def test_trace_id_unique(self):
        r1 = TraceRecord(query="a")
        r2 = TraceRecord(query="b")
        assert r1.trace_id != r2.trace_id

    def test_round_trip_minimal(self):
        record = TraceRecord(query="hello")
        restored = TraceRecord.model_validate_json(record.model_dump_json())
        assert restored.trace_id == record.trace_id
        assert restored.query == record.query

    def test_round_trip_full(self):
        record = TraceRecord(
            query="What is the capital of France?",
            retrieved_docs=[{"id": "doc1", "text": "Paris is the capital."}],
            reranked_docs=[{"id": "doc1", "score": 0.95}],
            prompt="Context: Paris is the capital.\n\nQ: What is the capital of France?",
            response="Paris.",
            latency_total_s=1.2,
            latency_llm_s=0.8,
            tokens=TokenUsage(prompt=120, completion=10, total=130),
            cost_usd=0.0001,
            spans=[
                SpanEvent(name="retriever", start_time=0.0, end_time=0.3, duration_s=0.3),
                SpanEvent(name="llm", start_time=0.4, end_time=1.2, duration_s=0.8),
            ],
            evaluation=EvalResult(faithfulness=0.95, context_relevance=0.88),
        )
        restored = TraceRecord.model_validate_json(record.model_dump_json())
        assert restored == record

    def test_json_is_valid_json(self):
        record = TraceRecord(query="test")
        parsed = json.loads(record.model_dump_json())
        assert "trace_id" in parsed
        assert "timestamp" in parsed

    def test_timestamp_is_utc(self):
        record = TraceRecord(query="test")
        assert record.timestamp.tzinfo == timezone.utc

    def test_error_field(self):
        record = TraceRecord(query="test", error="LLM timeout")
        restored = TraceRecord.model_validate_json(record.model_dump_json())
        assert restored.error == "LLM timeout"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            TraceRecord(query="test", unknown_field="x")  # type: ignore[call-arg]