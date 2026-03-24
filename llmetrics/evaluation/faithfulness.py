"""FaithfulnessEvaluator — cosine similarity of response sentences vs context."""
from __future__ import annotations

import re

import numpy as np

from llmetrics.core.config import settings
from llmetrics.core.models import EvalResult, TraceRecord
from llmetrics.evaluation.base import Evaluator


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FaithfulnessEvaluator(Evaluator):
    """
    Scores faithfulness as the fraction of response sentences whose cosine
    similarity to the concatenated reranked context exceeds the threshold.

    Embedding model is lazy-loaded on first evaluate() call.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold if threshold is not None else settings.faithfulness_threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def evaluate(self, record: TraceRecord) -> EvalResult:
        response = record.response
        context_parts = [
            str(d.get("content", "")) for d in record.reranked_docs if d.get("content")
        ]
        context = " ".join(context_parts)

        if not response or not context:
            return EvalResult()

        sentences = _split_sentences(response)
        if not sentences:
            return EvalResult()

        model = self._get_model()
        embeddings: np.ndarray = model.encode(sentences + [context])
        sent_embs = embeddings[: len(sentences)]
        ctx_emb = embeddings[len(sentences)]

        scores = [_cosine(s, ctx_emb) for s in sent_embs]
        faithfulness = sum(1 for s in scores if s >= self._threshold) / len(scores)
        return EvalResult(faithfulness=round(faithfulness, 4))