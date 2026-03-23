"""Span context manager — records wall-clock timing and appends to trace context."""
import time
from contextlib import contextmanager
from typing import Any, Generator

from llmetrics.core.models import SpanEvent
from llmetrics.tracing.context import append_span


@contextmanager
def span(name: str, metadata: dict[str, Any] | None = None) -> Generator[None, None, None]:
    """Context manager that times a block and records a SpanEvent."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        append_span(
            SpanEvent(
                name=name,
                start_time=start,
                end_time=end,
                duration_s=end - start,
                metadata=metadata or {},
            )
        )