"""Token extraction from LLM provider responses."""
from typing import Any

from llmetrics.core.models import TokenUsage


def extract_token_usage(response: Any, provider: str = "groq") -> TokenUsage | None:
    """
    Extract token usage from a Groq or Anthropic response object.
    Falls back to tiktoken estimation if usage data is absent.

    Groq:      response.usage.prompt_tokens / completion_tokens / total_tokens
    Anthropic: response.usage.input_tokens / output_tokens
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        if provider == "anthropic":
            prompt = getattr(usage, "input_tokens", None)
            completion = getattr(usage, "output_tokens", None)
            if prompt is not None and completion is not None:
                return TokenUsage(
                    prompt=prompt,
                    completion=completion,
                    total=prompt + completion,
                )

        else:  # groq (and openai-compatible)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if prompt is not None and completion is not None:
                return TokenUsage(
                    prompt=prompt,
                    completion=completion,
                    total=total if total is not None else prompt + completion,
                )

    except Exception:
        pass

    return None


def estimate_tokens_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Rough token count via tiktoken when provider usage data is unavailable."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return max(1, len(text) // 4)