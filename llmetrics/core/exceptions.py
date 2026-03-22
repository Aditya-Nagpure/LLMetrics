class LLMetricsError(Exception):
    """Base exception for all LLMetrics errors."""


class StorageError(LLMetricsError):
    """Raised when a storage read or write operation fails."""


class EvaluationError(LLMetricsError):
    """Raised when an evaluation step fails."""


class TracingError(LLMetricsError):
    """Raised when trace context is missing or corrupted."""


class PipelineError(LLMetricsError):
    """Raised when a RAG pipeline component fails."""


class ConfigurationError(LLMetricsError):
    """Raised when required configuration is missing or invalid."""