"""High-level Tracer convenience class."""
from contextlib import contextmanager
from typing import Any, Generator

from llmetrics.core.models import SpanEvent
from llmetrics.tracing import context as _ctx
from llmetrics.tracing.span import span as _span


class Tracer:
    def start_trace(self, trace_id: str | None = None) -> str:
        return _ctx.start_trace(trace_id)

    def finish_trace(self) -> tuple[str, list[SpanEvent]]:
        return _ctx.finish_trace()

    @contextmanager
    def span(self, name: str, metadata: dict[str, Any] | None = None) -> Generator[None, None, None]:
        with _span(name, metadata):
            yield


tracer = Tracer()
