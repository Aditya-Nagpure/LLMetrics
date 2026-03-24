"""Cross-encoder reranker — scores and sorts documents, wrapped in a tracing span."""
from __future__ import annotations

from llmetrics.core.config import settings
from llmetrics.core.exceptions import PipelineError
from llmetrics.tracing.span import span


class CrossEncoderReranker:
    """Re-ranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str | None = None, top_n: int | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._top_n = top_n if top_n is not None else settings.reranker_top_n
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            # torch >= 2.4 is required but not available on Intel Macs.
            # Fall back to returning docs in original order (no reranking).
            import warnings
            warnings.warn(
                f"CrossEncoderReranker could not load model ({exc}). "
                "Reranking disabled — docs will be returned in retrieval order.",
                stacklevel=2,
            )

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        """Score every (query, doc) pair and return the top-n by descending score.
        Falls back to returning the first top_n docs if the model is unavailable."""
        if not docs:
            return []
        if self._model is None:
            return docs[: self._top_n]
        with span("reranker", {"num_docs": len(docs), "top_n": self._top_n}):
            try:
                pairs = [(query, doc) for doc in docs]
                scores: list[float] = self._model.predict(pairs).tolist()
                ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                return [doc for _, doc in ranked[: self._top_n]]
            except Exception as exc:
                raise PipelineError(f"CrossEncoderReranker.rerank failed: {exc}") from exc