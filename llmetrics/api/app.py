"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from llmetrics.api.middleware import TracingMiddleware
from llmetrics.api.routes.metrics import router as metrics_router
from llmetrics.api.routes.query import router as query_router
from llmetrics.core.config import settings
from llmetrics.metrics.latency import LatencyTracker
from llmetrics.storage.base import StorageBackend


def _build_storage() -> StorageBackend:
    if settings.storage_backend == "jsonl":
        from llmetrics.storage.jsonl_store import JsonlStore

        return JsonlStore(settings.jsonl_path)
    from llmetrics.storage.sqlite_store import SqliteStore

    return SqliteStore(settings.sqlite_path)


def create_app(
    storage: StorageBackend | None = None,
    retriever: Any = None,
    reranker: Any = None,
    prompt_builder: Any = None,
    llm_client: Any = None,
    sync_evaluators: list[Any] | None = None,
    llm_judge: Any = None,
) -> FastAPI:
    """
    Build and return the FastAPI app.

    All pipeline components are optional — pass them in tests to inject mocks.
    When omitted, real implementations are constructed from settings at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ------ Startup ------
        app.state.storage = storage if storage is not None else _build_storage()
        app.state.latency_tracker = LatencyTracker()

        if retriever is not None:
            app.state.retriever = retriever
        else:
            from llmetrics.pipeline.retriever import ChromaRetriever

            app.state.retriever = ChromaRetriever()

        if reranker is not None:
            app.state.reranker = reranker
        else:
            from llmetrics.pipeline.reranker import CrossEncoderReranker

            app.state.reranker = CrossEncoderReranker()

        if prompt_builder is not None:
            app.state.prompt_builder = prompt_builder
        else:
            from llmetrics.pipeline.prompt_builder import PromptBuilder

            app.state.prompt_builder = PromptBuilder()

        if llm_client is not None:
            app.state.llm_client = llm_client
        else:
            from llmetrics.pipeline.llm_client import LLMClient

            app.state.llm_client = LLMClient()

        # ------ Evaluators ------
        if sync_evaluators is not None:
            app.state.sync_evaluators = sync_evaluators
        else:
            from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator
            from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

            app.state.sync_evaluators = [FaithfulnessEvaluator(), ContextRelevanceEvaluator()]

        if llm_judge is not None:
            app.state.llm_judge = llm_judge
        elif settings.enable_llm_judge:
            from llmetrics.evaluation.llm_judge import LLMJudgeEvaluator

            app.state.llm_judge = LLMJudgeEvaluator()
        else:
            app.state.llm_judge = None

        yield

        # ------ Shutdown ------
        app.state.storage.close()

    app = FastAPI(title="LLMetrics", version="0.1.0", lifespan=lifespan)
    app.add_middleware(TracingMiddleware)
    app.include_router(query_router)
    app.include_router(metrics_router)
    return app
