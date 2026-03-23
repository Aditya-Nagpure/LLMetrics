"""Ring-buffer latency tracker with P50/P95 percentiles."""
import statistics
from collections import deque


class LatencyTracker:
    """Thread-safe ring buffer for request latencies. Computes P50 and P95."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._buffer: deque[float] = deque(maxlen=maxlen)

    def record(self, latency_s: float) -> None:
        self._buffer.append(latency_s)

    def p50(self) -> float | None:
        return self._percentile(0.50)

    def p95(self) -> float | None:
        return self._percentile(0.95)

    def count(self) -> int:
        return len(self._buffer)

    def _percentile(self, q: float) -> float | None:
        data = list(self._buffer)
        if not data:
            return None
        if len(data) == 1:
            return data[0]
        return statistics.quantiles(data, n=100)[int(q * 100) - 1]