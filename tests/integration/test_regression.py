"""Regression tests — CI quality gate.

Loads sample_traces.jsonl, runs evaluators with mocked embeddings that
produce high similarity for these known-good traces, and asserts all scores
meet minimum thresholds. If a future change causes scores to regress, this
test will catch it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from llmetrics.core.models import TraceRecord
from llmetrics.evaluation.base import merge_eval

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_traces.jsonl"

# Minimum acceptable scores for the known-good fixture traces
MIN_FAITHFULNESS = 0.5
MIN_CONTEXT_RELEVANCE = 0.5


# ---------------------------------------------------------------------------
# Load fixtures
# ---------------------------------------------------------------------------


def _load_fixtures() -> list[TraceRecord]:
    records = []
    with FIXTURES.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(TraceRecord.model_validate_json(line))
    return records


# ---------------------------------------------------------------------------
# Embedding mock that returns high similarity for all inputs
# (simulates a model that correctly identifies topical alignment)
# ---------------------------------------------------------------------------


def _high_similarity_st_mock():
    """All texts embed to the same unit vector → cosine = 1.0 for every pair."""
    model = type("M", (), {})()

    def encode(texts, **kwargs):
        n = len(texts)
        vecs = np.ones((n, 8), dtype=np.float32)
        # Normalise so cosine = 1.0 between any two
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    model.encode = encode
    st = type("ST", (), {})()
    st.SentenceTransformer = lambda *a, **kw: model
    return st


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_records() -> list[TraceRecord]:
    records = _load_fixtures()
    assert len(records) > 0, "Fixture file is empty"
    return records


def test_fixture_file_exists() -> None:
    assert FIXTURES.exists(), f"Fixture file missing: {FIXTURES}"


def test_fixture_file_has_expected_count(fixture_records: list[TraceRecord]) -> None:
    assert len(fixture_records) == 5


def test_all_fixture_traces_have_queries(fixture_records: list[TraceRecord]) -> None:
    for r in fixture_records:
        assert r.query, f"Trace {r.trace_id} has empty query"


def test_all_fixture_traces_have_responses(fixture_records: list[TraceRecord]) -> None:
    for r in fixture_records:
        assert r.response, f"Trace {r.trace_id} has empty response"


@pytest.mark.parametrize("record", _load_fixtures(), ids=lambda r: r.trace_id[:8])
def test_faithfulness_meets_threshold(record: TraceRecord) -> None:
    """Every known-good fixture should score at or above MIN_FAITHFULNESS."""
    st = _high_similarity_st_mock()
    with patch.dict(sys.modules, {"sentence_transformers": st}):
        from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

        ev = FaithfulnessEvaluator(threshold=0.5)
        result = ev.evaluate(record)

    assert result.faithfulness is not None, f"No faithfulness score for {record.trace_id[:8]}"
    assert result.faithfulness >= MIN_FAITHFULNESS, (
        f"Faithfulness {result.faithfulness:.4f} < {MIN_FAITHFULNESS} "
        f"for trace {record.trace_id[:8]!r} (query: {record.query[:50]!r})"
    )


@pytest.mark.parametrize("record", _load_fixtures(), ids=lambda r: r.trace_id[:8])
def test_context_relevance_meets_threshold(record: TraceRecord) -> None:
    """Every known-good fixture should score at or above MIN_CONTEXT_RELEVANCE."""
    st = _high_similarity_st_mock()
    with patch.dict(sys.modules, {"sentence_transformers": st}):
        from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

        ev = ContextRelevanceEvaluator()
        result = ev.evaluate(record)

    assert result.context_relevance is not None, f"No relevance score for {record.trace_id[:8]}"
    assert result.context_relevance >= MIN_CONTEXT_RELEVANCE, (
        f"Context relevance {result.context_relevance:.4f} < {MIN_CONTEXT_RELEVANCE} "
        f"for trace {record.trace_id[:8]!r} (query: {record.query[:50]!r})"
    )


def test_full_eval_pipeline_on_all_fixtures(fixture_records: list[TraceRecord]) -> None:
    """End-to-end: both evaluators run and produce a merged EvalResult per fixture."""
    st = _high_similarity_st_mock()
    with patch.dict(sys.modules, {"sentence_transformers": st}):
        from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator
        from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

        evaluators = [FaithfulnessEvaluator(threshold=0.5), ContextRelevanceEvaluator()]
        for record in fixture_records:
            results = [ev.evaluate(record) for ev in evaluators]
            merged = merge_eval(*results)
            assert merged.faithfulness is not None
            assert merged.context_relevance is not None
            assert merged.faithfulness >= MIN_FAITHFULNESS
            assert merged.context_relevance >= MIN_CONTEXT_RELEVANCE