"""Async-safe request context using contextvars."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from llmetrics.core.models import SpanEvent


@dataclass
class _TraceContext:
    trace_id: str
    spans: list[SpanEvent] = field(default_factory=list)


_current: ContextVar[_TraceContext | None] = ContextVar("_llmetrics_trace", default=None)


def start_trace(trace_id: str | None = None) -> str:
    """Create a new trace context for the current async task. Returns the trace_id."""
    tid = trace_id or str(uuid4())
    _current.set(_TraceContext(trace_id=tid))
    return tid


def get_context() -> _TraceContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("No active trace context. Call start_trace() first.")
    return ctx


def append_span(span: SpanEvent) -> None:
    get_context().spans.append(span)


def finish_trace() -> tuple[str, list[SpanEvent]]:
    """Return (trace_id, spans) and clear the context."""
    ctx = get_context()
    _current.set(None)
    return ctx.trace_id, ctx.spans