"""Unit tests for Phase 6 evaluation engine.

All sentence-transformer calls are mocked via sys.modules so no real model
or GPU is needed. Embeddings are hand-crafted numpy arrays whose cosine
similarities are predictable.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llmetrics.core.models import EvalResult, TraceRecord
from llmetrics.evaluation.base import merge_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_DOC = [{"content": "Paris is the capital of France."}]


def _make_record(
    query: str = "What is the capital of France?",
    response: str = "Paris is the capital of France.",
    retrieved_docs: list[dict] | None = None,
    reranked_docs: list[dict] | None = None,
) -> TraceRecord:
    return TraceRecord(
        query=query,
        response=response,
        retrieved_docs=_DEFAULT_DOC if retrieved_docs is None else retrieved_docs,
        reranked_docs=_DEFAULT_DOC if reranked_docs is None else reranked_docs,
    )


def _st_mock(encode_return: np.ndarray) -> MagicMock:
    """Build a sys.modules-compatible sentence_transformers mock."""
    model = MagicMock()
    model.encode.return_value = encode_return
    st = MagicMock()
    st.SentenceTransformer.return_value = model
    return st


# ---------------------------------------------------------------------------
# merge_eval
# ---------------------------------------------------------------------------


class TestMergeEval:
    def test_merges_non_none_fields(self) -> None:
        a = EvalResult(faithfulness=0.9)
        b = EvalResult(context_relevance=0.7)
        merged = merge_eval(a, b)
        assert merged.faithfulness == 0.9
        assert merged.context_relevance == 0.7
        assert merged.answer_correctness is None

    def test_last_value_wins(self) -> None:
        a = EvalResult(faithfulness=0.5)
        b = EvalResult(faithfulness=0.9)
        merged = merge_eval(a, b)
        assert merged.faithfulness == 0.9

    def test_empty_results(self) -> None:
        merged = merge_eval(EvalResult(), EvalResult())
        assert merged.faithfulness is None
        assert merged.context_relevance is None


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator
# ---------------------------------------------------------------------------


class TestFaithfulnessEvaluator:
    def test_faithful_answer_scores_high(self) -> None:
        # sentence and context embed to the same vector → cosine = 1.0
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])  # [sentence, context]
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev = FaithfulnessEvaluator(threshold=0.5)
            result = ev.evaluate(_make_record())

        assert result.faithfulness == pytest.approx(1.0)

    def test_hallucinated_answer_scores_low(self) -> None:
        # sentence and context are orthogonal → cosine = 0.0 → below threshold
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev = FaithfulnessEvaluator(threshold=0.5)
            result = ev.evaluate(
                _make_record(response="The moon is made of cheese.", reranked_docs=[{"content": "Paris is in France."}])
            )

        assert result.faithfulness == pytest.approx(0.0)

    def test_faithful_scores_higher_than_hallucinated(self) -> None:
        # Faithful: cosine = 1.0; hallucinated: cosine = 0.0
        faithful_embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        hallucinated_embs = np.array([[1.0, 0.0], [0.0, 1.0]])

        with patch.dict(sys.modules, {"sentence_transformers": _st_mock(faithful_embs)}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev_faithful = FaithfulnessEvaluator(threshold=0.5)
            faithful_score = ev_faithful.evaluate(_make_record()).faithfulness

        with patch.dict(sys.modules, {"sentence_transformers": _st_mock(hallucinated_embs)}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev_hallucinated = FaithfulnessEvaluator(threshold=0.5)
            hallucinated_score = ev_hallucinated.evaluate(_make_record()).faithfulness

        assert faithful_score is not None
        assert hallucinated_score is not None
        assert faithful_score > hallucinated_score

    def test_empty_response_returns_empty_eval(self) -> None:
        st = _st_mock(np.array([]))

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev = FaithfulnessEvaluator()
            result = ev.evaluate(_make_record(response=""))

        assert result.faithfulness is None

    def test_empty_context_returns_empty_eval(self) -> None:
        st = _st_mock(np.array([]))

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev = FaithfulnessEvaluator()
            result = ev.evaluate(_make_record(reranked_docs=[]))

        assert result.faithfulness is None

    def test_multiple_sentences_partial_faithfulness(self) -> None:
        # 2 sentences: first is faithful (cosine=1.0), second is not (cosine=0.0)
        # threshold=0.5 → 1 of 2 sentences pass → faithfulness = 0.5
        embeddings = np.array([
            [1.0, 0.0],  # sentence 1 — faithful
            [0.0, 1.0],  # sentence 2 — hallucinated
            [1.0, 0.0],  # context
        ])
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            ev = FaithfulnessEvaluator(threshold=0.5)
            result = ev.evaluate(
                _make_record(response="First sentence. Second sentence.")
            )

        assert result.faithfulness == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ContextRelevanceEvaluator
# ---------------------------------------------------------------------------


class TestContextRelevanceEvaluator:
    def test_relevant_docs_score_high(self) -> None:
        # query and doc embed to same vector → cosine = 1.0
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev = ContextRelevanceEvaluator()
            result = ev.evaluate(_make_record())

        assert result.context_relevance == pytest.approx(1.0)

    def test_irrelevant_docs_score_low(self) -> None:
        # query and doc are orthogonal → cosine = 0.0
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev = ContextRelevanceEvaluator()
            result = ev.evaluate(_make_record())

        assert result.context_relevance == pytest.approx(0.0)

    def test_relevant_scores_higher_than_irrelevant(self) -> None:
        relevant_embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        irrelevant_embs = np.array([[1.0, 0.0], [0.0, 1.0]])

        with patch.dict(sys.modules, {"sentence_transformers": _st_mock(relevant_embs)}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev = ContextRelevanceEvaluator()
            relevant_score = ev.evaluate(_make_record()).context_relevance

        with patch.dict(sys.modules, {"sentence_transformers": _st_mock(irrelevant_embs)}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev2 = ContextRelevanceEvaluator()
            irrelevant_score = ev2.evaluate(_make_record()).context_relevance

        assert relevant_score is not None
        assert irrelevant_score is not None
        assert relevant_score > irrelevant_score

    def test_no_docs_returns_empty_eval(self) -> None:
        st = _st_mock(np.array([]))

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev = ContextRelevanceEvaluator()
            result = ev.evaluate(_make_record(retrieved_docs=[]))

        assert result.context_relevance is None

    def test_mean_of_multiple_docs(self) -> None:
        # query=[1,0], doc1=[1,0] (cos=1.0), doc2=[0,1] (cos=0.0) → mean=0.5
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        st = _st_mock(embeddings)

        with patch.dict(sys.modules, {"sentence_transformers": st}):
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator

            ev = ContextRelevanceEvaluator()
            result = ev.evaluate(
                _make_record(
                    retrieved_docs=[{"content": "doc1"}, {"content": "doc2"}]
                )
            )

        assert result.context_relevance == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator
# ---------------------------------------------------------------------------


class TestLLMJudgeEvaluator:
    def _make_groq_response(self, json_text: str) -> MagicMock:
        resp = MagicMock()
        resp.choices[0].message.content = json_text
        return resp

    def test_parses_groq_json_response(self) -> None:
        fake_resp = self._make_groq_response(
            '{"faithfulness": 0.9, "context_relevance": 0.8, "answer_correctness": 0.85}'
        )

        with patch("groq.Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = fake_resp

            from llmetrics.evaluation.llm_judge import LLMJudgeEvaluator

            ev = LLMJudgeEvaluator()
            result = ev.evaluate(_make_record())

        assert result.faithfulness == pytest.approx(0.9)
        assert result.context_relevance == pytest.approx(0.8)
        assert result.answer_correctness == pytest.approx(0.85)

    def test_returns_empty_eval_on_bad_response(self) -> None:
        fake_resp = self._make_groq_response("I cannot evaluate this.")

        with patch("groq.Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = fake_resp

            from llmetrics.evaluation.llm_judge import LLMJudgeEvaluator

            ev = LLMJudgeEvaluator()
            result = ev.evaluate(_make_record())

        assert result.faithfulness is None

    def test_tolerates_json_embedded_in_prose(self) -> None:
        text = 'Sure! Here are the scores: {"faithfulness": 0.7, "context_relevance": 0.6, "answer_correctness": 0.8} Hope that helps!'
        fake_resp = self._make_groq_response(text)

        with patch("groq.Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = fake_resp

            from llmetrics.evaluation.llm_judge import LLMJudgeEvaluator

            ev = LLMJudgeEvaluator()
            result = ev.evaluate(_make_record())

        assert result.faithfulness == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_evaluate_async_updates_storage(self) -> None:
        fake_resp = self._make_groq_response(
            '{"faithfulness": 0.95, "context_relevance": 0.9, "answer_correctness": 0.88}'
        )
        storage = MagicMock()
        record = _make_record()

        with patch("groq.Groq") as mock_groq:
            mock_groq.return_value.chat.completions.create.return_value = fake_resp

            from llmetrics.evaluation.llm_judge import LLMJudgeEvaluator

            ev = LLMJudgeEvaluator()
            await ev.evaluate_async(record, storage)

        storage.write.assert_called_once()
        written: TraceRecord = storage.write.call_args[0][0]
        assert written.evaluation.faithfulness == pytest.approx(0.95)
        assert written.trace_id == record.trace_id