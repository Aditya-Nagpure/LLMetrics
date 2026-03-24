"""LLM client — dispatches to Groq or Anthropic, wrapped in a tracing span."""
from __future__ import annotations

import time
from typing import Any

from llmetrics.core.config import settings
from llmetrics.core.exceptions import PipelineError
from llmetrics.core.models import TokenUsage
from llmetrics.metrics.cost import estimate_cost
from llmetrics.metrics.tokens import extract_token_usage
from llmetrics.tracing.span import span


class LLMClient:
    """Sends prompts to the configured LLM provider and records a 'llm' span."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self._provider = provider or settings.llm_provider
        self._model = model or settings.llm_model
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if self._provider == "groq":
            from groq import Groq

            return Groq(api_key=settings.groq_api_key)
        elif self._provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            raise PipelineError(f"Unsupported LLM provider: {self._provider!r}")

    def complete(self, prompt: str) -> tuple[str, TokenUsage | None, float]:
        """
        Send *prompt* to the LLM and return (response_text, token_usage, latency_s).

        The wall-clock time covers only the network round-trip; token extraction
        and span recording happen outside that window.
        """
        with span("llm", {"provider": self._provider, "model": self._model}):
            start = time.perf_counter()
            try:
                response_text, raw_response = self._call(prompt)
            except Exception as exc:
                raise PipelineError(f"LLMClient.complete failed: {exc}") from exc
            latency = time.perf_counter() - start

            tokens = extract_token_usage(raw_response, self._provider)

        return response_text, tokens, latency

    def _call(self, prompt: str) -> tuple[str, Any]:
        if self._provider == "groq":
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content, response

        # anthropic
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text, response

    def estimate_cost(self, tokens: TokenUsage) -> float:
        """Convenience wrapper around the cost module."""
        return estimate_cost(
            tokens,
            model=self._model,
            cost_input_per_1m=settings.cost_input_per_1m,
            cost_output_per_1m=settings.cost_output_per_1m,
        )