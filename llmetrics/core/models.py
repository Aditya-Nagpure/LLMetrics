from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: int
    completion: int
    total: int


class SpanEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    start_time: float
    end_time: float
    duration_s: float
    metadata: dict[str, object] = Field(default_factory=dict)


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: float | None = None
    context_relevance: float | None = None
    answer_correctness: float | None = None


class TraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Request
    query: str

    # Pipeline stages
    retrieved_docs: list[dict[str, object]] = Field(default_factory=list)
    reranked_docs: list[dict[str, object]] = Field(default_factory=list)
    prompt: str = ""
    response: str = ""

    # Metrics
    latency_total_s: float = 0.0
    latency_llm_s: float = 0.0
    tokens: TokenUsage | None = None
    cost_usd: float = 0.0

    # Tracing
    spans: list[SpanEvent] = Field(default_factory=list)

    # Evaluation
    evaluation: EvalResult = Field(default_factory=EvalResult)

    # Error capture
    error: str | None = None