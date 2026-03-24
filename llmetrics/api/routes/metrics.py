"""GET /metrics — returns in-process latency percentiles and request count."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from llmetrics.api.dependencies import get_latency_tracker

router = APIRouter()


class MetricsResponse(BaseModel):
    p50_latency_s: float | None
    p95_latency_s: float | None
    total_requests: int


@router.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint(
    tracker=Depends(get_latency_tracker),
) -> MetricsResponse:
    return MetricsResponse(
        p50_latency_s=tracker.p50(),
        p95_latency_s=tracker.p95(),
        total_requests=tracker.count(),
    )
