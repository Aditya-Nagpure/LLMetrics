"""POST /query — orchestrates the full RAG pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from llmetrics.api.dependencies import (
    get_llm_client,
    get_prompt_builder,
    get_reranker,
    get_retriever,
)

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    trace_id: str


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    prompt_builder=Depends(get_prompt_builder),
    llm_client=Depends(get_llm_client),
) -> QueryResponse:
    # Retrieve
    retrieved: list[str] = retriever.retrieve(body.query)
    request.state.retrieved_docs = [{"content": d} for d in retrieved]

    # Rerank
    reranked: list[str] = reranker.rerank(body.query, retrieved)
    request.state.reranked_docs = [{"content": d} for d in reranked]

    # Build prompt
    prompt: str = prompt_builder.build(body.query, reranked)
    request.state.prompt = prompt

    # LLM call
    response_text, tokens, latency_llm = llm_client.complete(prompt)
    request.state.response = response_text
    request.state.tokens = tokens
    request.state.latency_llm_s = latency_llm

    return QueryResponse(response=response_text, trace_id=request.state.trace_id)
