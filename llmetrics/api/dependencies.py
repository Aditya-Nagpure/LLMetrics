"""FastAPI Depends factories — pull singletons from app.state."""
from __future__ import annotations

from fastapi import Request

from llmetrics.metrics.latency import LatencyTracker
from llmetrics.storage.base import StorageBackend


def get_storage(request: Request) -> StorageBackend:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_latency_tracker(request: Request) -> LatencyTracker:
    return request.app.state.latency_tracker  # type: ignore[no-any-return]


def get_retriever(request: Request):  # type: ignore[return]
    return request.app.state.retriever


def get_reranker(request: Request):  # type: ignore[return]
    return request.app.state.reranker


def get_prompt_builder(request: Request):  # type: ignore[return]
    return request.app.state.prompt_builder


def get_llm_client(request: Request):  # type: ignore[return]
    return request.app.state.llm_client
