"""TracingMiddleware — manages the full trace lifecycle for every /query request."""
from __future__ import annotations

import json
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from llmetrics.core.config import settings
from llmetrics.core.models import TraceRecord
from llmetrics.metrics.cost import estimate_cost
from llmetrics.tracing.context import finish_trace, start_trace


class TracingMiddleware(BaseHTTPMiddleware):
    """
    For every POST /query request:

    1. Reads and caches the request body (so the query is captured even if the
       route handler raises before it can set request.state.query).
    2. Calls start_trace() to create an async-safe context.
    3. Calls the next handler and measures wall-clock latency.
    4. In the finally block, assembles a TraceRecord from request.state and
       writes it to storage, then records the latency.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        is_query = request.method == "POST" and request.url.path == "/query"

        trace_id = start_trace()
        request.state.trace_id = trace_id

        if is_query:
            # Cache the body so the route handler can still read it via FastAPI
            # and we have the query for the trace even if the handler crashes.
            try:
                raw = await request.body()
                request.state.query = json.loads(raw).get("query", "")
            except Exception:
                request.state.query = ""

            # Defaults — overwritten by the route handler on success
            request.state.retrieved_docs = []
            request.state.reranked_docs = []
            request.state.prompt = ""
            request.state.response = ""
            request.state.tokens = None
            request.state.latency_llm_s = 0.0
            request.state.error = None

        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            total_latency = time.perf_counter() - start
            _, spans = finish_trace()

            if is_query:
                tokens = getattr(request.state, "tokens", None)
                cost = 0.0
                if tokens is not None:
                    cost = estimate_cost(
                        tokens,
                        model=settings.llm_model,
                        cost_input_per_1m=settings.cost_input_per_1m,
                        cost_output_per_1m=settings.cost_output_per_1m,
                    )

                record = TraceRecord(
                    trace_id=trace_id,
                    query=getattr(request.state, "query", ""),
                    retrieved_docs=getattr(request.state, "retrieved_docs", []),
                    reranked_docs=getattr(request.state, "reranked_docs", []),
                    prompt=getattr(request.state, "prompt", ""),
                    response=getattr(request.state, "response", ""),
                    latency_total_s=total_latency,
                    latency_llm_s=getattr(request.state, "latency_llm_s", 0.0),
                    tokens=tokens,
                    cost_usd=cost,
                    spans=spans,
                    error=getattr(request.state, "error", None),
                )

                try:
                    request.app.state.storage.write(record)
                    request.app.state.latency_tracker.record(total_latency)
                except Exception:
                    pass  # Never let observability break the response
