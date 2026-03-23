"""Unit tests for Phase 3: tracing context, span timing, latency tracker, cost, tokens."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llmetrics.core.models import TokenUsage
from llmetrics.metrics.cost import PRICING, estimate_cost
from llmetrics.metrics.latency import LatencyTracker
from llmetrics.metrics.tokens import extract_token_usage
from llmetrics.tracing.context import finish_trace, get_context, start_trace
from llmetrics.tracing.span import span
from llmetrics.tracing.tracer import Tracer


# ---------------------------------------------------------------------------
# Tracing context
# ---------------------------------------------------------------------------

class TestTraceContext:
    def setup_method(self):
        # Ensure clean context before each test
        try:
            finish_trace()
        except RuntimeError:
            pass

    def test_start_trace_returns_id(self):
        tid = start_trace()
        assert isinstance(tid, str) and len(tid) > 0
        finish_trace()

    def test_explicit_trace_id(self):
        tid = start_trace("my-trace-123")
        assert get_context().trace_id == "my-trace-123"
        finish_trace()

    def test_no_context_raises(self):
        with pytest.raises(RuntimeError, match="No active trace context"):
            get_context()

    def test_finish_returns_id_and_spans(self):
        tid = start_trace()
        trace_id, spans = finish_trace()
        assert trace_id == tid
        assert spans == []

    def test_finish_clears_context(self):
        start_trace()
        finish_trace()
        with pytest.raises(RuntimeError):
            get_context()

    def test_span_appended_to_context(self):
        start_trace()
        with span("test_span"):
            pass
        _, spans = finish_trace()
        assert len(spans) == 1
        assert spans[0].name == "test_span"

    def test_span_timing_positive(self):
        start_trace()
        with span("timed"):
            pass
        _, spans = finish_trace()
        assert spans[0].duration_s >= 0.0
        assert spans[0].end_time >= spans[0].start_time

    def test_span_metadata(self):
        start_trace()
        with span("meta_span", metadata={"model": "llama3"}):
            pass
        _, spans = finish_trace()
        assert spans[0].metadata == {"model": "llama3"}

    def test_multiple_spans(self):
        start_trace()
        with span("a"):
            pass
        with span("b"):
            pass
        with span("c"):
            pass
        _, spans = finish_trace()
        assert [s.name for s in spans] == ["a", "b", "c"]

    def test_span_records_even_on_exception(self):
        start_trace()
        try:
            with span("error_span"):
                raise ValueError("boom")
        except ValueError:
            pass
        _, spans = finish_trace()
        assert len(spans) == 1
        assert spans[0].name == "error_span"


class TestTracerClass:
    def setup_method(self):
        try:
            finish_trace()
        except RuntimeError:
            pass

    def test_tracer_start_finish(self):
        t = Tracer()
        tid = t.start_trace()
        trace_id, spans = t.finish_trace()
        assert tid == trace_id
        assert spans == []

    def test_tracer_span(self):
        t = Tracer()
        t.start_trace()
        with t.span("pipeline"):
            pass
        _, spans = t.finish_trace()
        assert spans[0].name == "pipeline"


class TestAsyncTraceIsolation:
    """Verify contextvars isolate traces across concurrent async tasks."""

    async def _run_trace(self, name: str) -> tuple[str, list]:
        tid = start_trace(name)
        with span(f"span_{name}"):
            await asyncio.sleep(0)  # yield to event loop
        return finish_trace()

    @pytest.mark.asyncio
    async def test_concurrent_tasks_isolated(self):
        results = await asyncio.gather(
            self._run_trace("trace_A"),
            self._run_trace("trace_B"),
            self._run_trace("trace_C"),
        )
        ids = [r[0] for r in results]
        assert ids == ["trace_A", "trace_B", "trace_C"]
        for trace_id, spans in results:
            assert len(spans) == 1
            assert spans[0].name == f"span_{trace_id}"


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class TestLatencyTracker:
    def test_empty_returns_none(self):
        t = LatencyTracker()
        assert t.p50() is None
        assert t.p95() is None

    def test_single_value(self):
        t = LatencyTracker()
        t.record(1.0)
        assert t.p50() == 1.0
        assert t.p95() == 1.0

    def test_p50_median(self):
        t = LatencyTracker()
        for v in range(1, 101):  # 1..100
            t.record(float(v))
        p50 = t.p50()
        assert p50 is not None
        assert 49.0 <= p50 <= 51.0

    def test_p95_tail(self):
        t = LatencyTracker()
        for v in range(1, 101):
            t.record(float(v))
        p95 = t.p95()
        assert p95 is not None
        assert p95 >= 90.0

    def test_p95_greater_than_p50(self):
        t = LatencyTracker()
        for v in range(1, 101):
            t.record(float(v))
        assert t.p95() > t.p50()  # type: ignore[operator]

    def test_ring_buffer_evicts_old(self):
        t = LatencyTracker(maxlen=10)
        for _ in range(5):
            t.record(100.0)  # old, high latency values
        for _ in range(10):
            t.record(1.0)   # new, low — should evict the old ones
        assert t.count() == 10
        p95 = t.p95()
        assert p95 is not None
        assert p95 < 50.0  # dominated by 1.0 values

    def test_count(self):
        t = LatencyTracker()
        for _ in range(7):
            t.record(0.1)
        assert t.count() == 7


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

class TestExtractTokenUsage:
    def _groq_response(self, prompt=100, completion=50, total=150):
        usage = SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
        )
        return SimpleNamespace(usage=usage)

    def _anthropic_response(self, input_tokens=100, output_tokens=50):
        usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        return SimpleNamespace(usage=usage)

    def test_groq_extraction(self):
        result = extract_token_usage(self._groq_response(), provider="groq")
        assert result == TokenUsage(prompt=100, completion=50, total=150)

    def test_anthropic_extraction(self):
        result = extract_token_usage(self._anthropic_response(), provider="anthropic")
        assert result == TokenUsage(prompt=100, completion=50, total=150)

    def test_anthropic_total_computed(self):
        result = extract_token_usage(self._anthropic_response(80, 20), provider="anthropic")
        assert result is not None
        assert result.total == 100

    def test_no_usage_attribute_returns_none(self):
        result = extract_token_usage(SimpleNamespace(), provider="groq")
        assert result is None

    def test_missing_fields_returns_none(self):
        response = SimpleNamespace(usage=SimpleNamespace())
        result = extract_token_usage(response, provider="groq")
        assert result is None


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        tokens = TokenUsage(prompt=1_000_000, completion=1_000_000, total=2_000_000)
        input_price, output_price = PRICING["llama3-8b-8192"]
        expected = (input_price + output_price) / 1  # 1M each
        assert estimate_cost(tokens, model="llama3-8b-8192") == pytest.approx(expected, rel=1e-6)

    def test_default_fallback(self):
        tokens = TokenUsage(prompt=500_000, completion=500_000, total=1_000_000)
        cost = estimate_cost(tokens, model="unknown-model-xyz")
        input_p, output_p = PRICING["default"]
        expected = (500_000 * input_p + 500_000 * output_p) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_override_prices(self):
        tokens = TokenUsage(prompt=1_000_000, completion=0, total=1_000_000)
        cost = estimate_cost(tokens, model="llama3-8b-8192", cost_input_per_1m=10.0)
        assert cost == pytest.approx(10.0)

    def test_zero_tokens(self):
        tokens = TokenUsage(prompt=0, completion=0, total=0)
        assert estimate_cost(tokens) == 0.0

    def test_cost_scales_linearly(self):
        t1 = TokenUsage(prompt=1000, completion=1000, total=2000)
        t2 = TokenUsage(prompt=2000, completion=2000, total=4000)
        assert estimate_cost(t2) == pytest.approx(estimate_cost(t1) * 2)

    def test_anthropic_model_priced_higher_than_groq(self):
        tokens = TokenUsage(prompt=10_000, completion=10_000, total=20_000)
        groq_cost = estimate_cost(tokens, model="llama3-8b-8192")
        claude_cost = estimate_cost(tokens, model="claude-sonnet-4-6")
        assert claude_cost > groq_cost