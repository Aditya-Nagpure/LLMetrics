"""ChromaDB retriever — wraps document lookup in a tracing span."""
from __future__ import annotations

import chromadb

from llmetrics.core.config import settings
from llmetrics.core.exceptions import PipelineError
from llmetrics.tracing.span import span


class ChromaRetriever:
    """Retrieves documents from ChromaDB and emits a 'retriever' span."""

    def __init__(
        self,
        collection_name: str | None = None,
        top_k: int | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._collection_name = collection_name or settings.chroma_collection
        self._top_k = top_k if top_k is not None else settings.retriever_top_k
        try:
            client = chromadb.HttpClient(
                host=host or settings.chroma_host,
                port=port or settings.chroma_port,
            )
            self._collection = client.get_collection(self._collection_name)
        except Exception as exc:
            raise PipelineError(f"ChromaRetriever init failed: {exc}") from exc

    def retrieve(self, query: str) -> list[str]:
        """Query ChromaDB and return the top-k document strings."""
        with span("retriever", {"query": query[:200], "top_k": self._top_k}):
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=self._top_k,
                )
                docs: list[str] = results["documents"][0] if results.get("documents") else []
                return docs
            except Exception as exc:
                raise PipelineError(f"ChromaRetriever.retrieve failed: {exc}") from exc