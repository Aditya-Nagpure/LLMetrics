"""Integration tests for Phase 4 pipeline components.

Each test:
  1. Starts a fresh trace context so spans can be recorded.
  2. Exercises the component under test with mocked external dependencies.
  3. Finishes the trace and asserts the expected span name was emitted.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llmetrics.core.models import TokenUsage
from llmetrics.tracing.context import finish_trace, start_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span_names(spans: list) -> list[str]:
    return [s.name for s in spans]


# ---------------------------------------------------------------------------
# ChromaRetriever
# ---------------------------------------------------------------------------


class TestChromaRetriever:
    def test_retrieve_returns_documents(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [["doc_a", "doc_b"]]}

        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value.get_collection.return_value = mock_collection

            from llmetrics.pipeline.retriever import ChromaRetriever

            retriever = ChromaRetriever(collection_name="test", top_k=2)
            start_trace()
            docs = retriever.retrieve("what is RAG?")
            _, spans = finish_trace()

        assert docs == ["doc_a", "doc_b"]
        assert "retriever" in _span_names(spans)

    def test_retrieve_emits_single_span(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [["only_doc"]]}

        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value.get_collection.return_value = mock_collection

            from llmetrics.pipeline.retriever import ChromaRetriever

            retriever = ChromaRetriever(collection_name="test", top_k=1)
            start_trace()
            retriever.retrieve("query")
            _, spans = finish_trace()

        retriever_spans = [s for s in spans if s.name == "retriever"]
        assert len(retriever_spans) == 1

    def test_retrieve_empty_results(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]]}

        with patch("chromadb.HttpClient") as mock_http:
            mock_http.return_value.get_collection.return_value = mock_collection

            from llmetrics.pipeline.retriever import ChromaRetriever

            retriever = ChromaRetriever(collection_name="test", top_k=5)
            start_trace()
            docs = retriever.retrieve("unknown query")
            _, spans = finish_trace()

        assert docs == []
        assert "retriever" in _span_names(spans)


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


def _make_st_mock(model_instance: MagicMock) -> MagicMock:
    """Build a fake sentence_transformers sys.modules entry."""
    st_mock = MagicMock()
    st_mock.CrossEncoder.return_value = model_instance
    return st_mock


class TestCrossEncoderReranker:
    def test_rerank_orders_by_score(self) -> None:
        mock_model = MagicMock()
        # scores: doc_b (0.9) > doc_a (0.1) — reranked order should be [doc_b, doc_a]
        mock_model.predict.return_value = np.array([0.1, 0.9])

        with patch.dict(sys.modules, {"sentence_transformers": _make_st_mock(mock_model)}):
            from llmetrics.pipeline.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name="test-model", top_n=2)
            start_trace()
            result = reranker.rerank("query", ["doc_a", "doc_b"])
            _, spans = finish_trace()

        assert result == ["doc_b", "doc_a"]
        assert "reranker" in _span_names(spans)

    def test_rerank_respects_top_n(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])

        with patch.dict(sys.modules, {"sentence_transformers": _make_st_mock(mock_model)}):
            from llmetrics.pipeline.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name="test-model", top_n=2)
            start_trace()
            result = reranker.rerank("query", ["a", "b", "c"])
            _, spans = finish_trace()

        assert len(result) == 2
        assert result[0] == "b"  # highest score

    def test_rerank_empty_docs_returns_empty(self) -> None:
        mock_model = MagicMock()

        with patch.dict(sys.modules, {"sentence_transformers": _make_st_mock(mock_model)}):
            from llmetrics.pipeline.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name="test-model", top_n=3)
            start_trace()
            result = reranker.rerank("query", [])
            _, spans = finish_trace()

        assert result == []
        # No span emitted when docs list is empty (early return before with-block)
        assert "reranker" not in _span_names(spans)

    def test_rerank_emits_span(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])

        with patch.dict(sys.modules, {"sentence_transformers": _make_st_mock(mock_model)}):
            from llmetrics.pipeline.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name="test-model", top_n=1)
            start_trace()
            reranker.rerank("q", ["single doc"])
            _, spans = finish_trace()

        reranker_spans = [s for s in spans if s.name == "reranker"]
        assert len(reranker_spans) == 1
        assert reranker_spans[0].metadata["num_docs"] == 1
        assert reranker_spans[0].metadata["top_n"] == 1


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    def test_build_includes_query_and_docs(self) -> None:
        from llmetrics.pipeline.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        start_trace()
        prompt = builder.build("What is LLM?", ["LLMs are large models.", "They use transformers."])
        _, spans = finish_trace()

        assert "What is LLM?" in prompt
        assert "LLMs are large models." in prompt
        assert "They use transformers." in prompt
        assert "prompt_builder" in _span_names(spans)

    def test_build_emits_single_span(self) -> None:
        from llmetrics.pipeline.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        start_trace()
        builder.build("q", ["doc1"])
        _, spans = finish_trace()

        pb_spans = [s for s in spans if s.name == "prompt_builder"]
        assert len(pb_spans) == 1
        assert pb_spans[0].metadata["num_docs"] == 1

    def test_build_custom_template(self) -> None:
        from llmetrics.pipeline.prompt_builder import PromptBuilder

        template = "Q: {{ query }}\nDocs: {% for doc in docs %}{{ doc }} {% endfor %}"
        builder = PromptBuilder(template=template)
        start_trace()
        prompt = builder.build("hello", ["doc_a", "doc_b"])
        finish_trace()

        assert prompt == "Q: hello\nDocs: doc_a doc_b "

    def test_build_empty_docs(self) -> None:
        from llmetrics.pipeline.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        start_trace()
        prompt = builder.build("query", [])
        _, spans = finish_trace()

        assert "query" in prompt
        assert "prompt_builder" in _span_names(spans)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class TestLLMClientGroq:
    def _make_groq_response(self, text: str) -> MagicMock:
        response = MagicMock()
        response.choices[0].message.content = text
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.usage.total_tokens = 15
        return response

    def test_complete_returns_text_and_tokens(self) -> None:
        fake_response = self._make_groq_response("answer text")

        with patch("groq.Groq") as mock_groq_cls:
            mock_client = mock_groq_cls.return_value
            mock_client.chat.completions.create.return_value = fake_response

            from llmetrics.pipeline.llm_client import LLMClient

            client = LLMClient(provider="groq", model="llama3-8b-8192")
            start_trace()
            text, tokens, latency = client.complete("prompt text")
            _, spans = finish_trace()

        assert text == "answer text"
        assert isinstance(tokens, TokenUsage)
        assert tokens.prompt == 10
        assert tokens.completion == 5
        assert tokens.total == 15
        assert latency >= 0.0
        assert "llm" in _span_names(spans)

    def test_complete_emits_llm_span_with_metadata(self) -> None:
        fake_response = self._make_groq_response("resp")

        with patch("groq.Groq") as mock_groq_cls:
            mock_groq_cls.return_value.chat.completions.create.return_value = fake_response

            from llmetrics.pipeline.llm_client import LLMClient

            client = LLMClient(provider="groq", model="llama3-8b-8192")
            start_trace()
            client.complete("p")
            _, spans = finish_trace()

        llm_spans = [s for s in spans if s.name == "llm"]
        assert len(llm_spans) == 1
        assert llm_spans[0].metadata["provider"] == "groq"
        assert llm_spans[0].metadata["model"] == "llama3-8b-8192"


class TestLLMClientAnthropic:
    def _make_anthropic_response(self, text: str) -> MagicMock:
        response = MagicMock()
        response.content[0].text = text
        response.usage.input_tokens = 20
        response.usage.output_tokens = 8
        return response

    def test_complete_returns_text_and_tokens(self) -> None:
        fake_response = self._make_anthropic_response("anthropic answer")

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = mock_anthropic_cls.return_value
            mock_client.messages.create.return_value = fake_response

            from llmetrics.pipeline.llm_client import LLMClient

            client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
            start_trace()
            text, tokens, latency = client.complete("my prompt")
            _, spans = finish_trace()

        assert text == "anthropic answer"
        assert isinstance(tokens, TokenUsage)
        assert tokens.prompt == 20
        assert tokens.completion == 8
        assert tokens.total == 28
        assert "llm" in _span_names(spans)

    def test_complete_span_metadata_provider_anthropic(self) -> None:
        fake_response = self._make_anthropic_response("r")

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_anthropic_cls.return_value.messages.create.return_value = fake_response

            from llmetrics.pipeline.llm_client import LLMClient

            client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
            start_trace()
            client.complete("p")
            _, spans = finish_trace()

        llm_spans = [s for s in spans if s.name == "llm"]
        assert llm_spans[0].metadata["provider"] == "anthropic"
        assert llm_spans[0].metadata["model"] == "claude-sonnet-4-6"

    def test_estimate_cost(self) -> None:
        fake_response = self._make_anthropic_response("x")

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_anthropic_cls.return_value.messages.create.return_value = fake_response

            from llmetrics.pipeline.llm_client import LLMClient

            client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
            start_trace()
            _, tokens, _ = client.complete("p")
            finish_trace()

        assert tokens is not None
        cost = client.estimate_cost(tokens)
        assert cost > 0.0