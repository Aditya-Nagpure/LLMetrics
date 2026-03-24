"""Abstract evaluator interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from llmetrics.core.models import EvalResult, TraceRecord


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, record: TraceRecord) -> EvalResult: ...


def merge_eval(*results: EvalResult) -> EvalResult:
    """Merge multiple EvalResults — last non-None value for each field wins."""
    data: dict[str, float | None] = {
        "faithfulness": None,
        "context_relevance": None,
        "answer_correctness": None,
    }
    for result in results:
        for k, v in result.model_dump().items():
            if v is not None:
                data[k] = v
    return EvalResult(**data)