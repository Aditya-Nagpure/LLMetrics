"""ContextRelevanceEvaluator — mean cosine similarity of query vs retrieved docs."""
from __future__ import annotations

import numpy as np

from llmetrics.core.models import EvalResult, TraceRecord
from llmetrics.evaluation.base import Evaluator
from llmetrics.evaluation.faithfulness import _cosine


class ContextRelevanceEvaluator(Evaluator):
    """
    Scores context relevance as the mean cosine similarity between the query
    embedding and each retrieved document embedding.

    Embedding model is lazy-loaded on first evaluate() call.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def evaluate(self, record: TraceRecord) -> EvalResult:
        query = record.query
        doc_texts = [
            str(d.get("content", "")) for d in record.retrieved_docs if d.get("content")
        ]

        if not query or not doc_texts:
            return EvalResult()

        model = self._get_model()
        embeddings: np.ndarray = model.encode([query] + doc_texts)
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        scores = [_cosine(query_emb, d) for d in doc_embs]
        relevance = sum(scores) / len(scores)
        return EvalResult(context_relevance=round(relevance, 4))