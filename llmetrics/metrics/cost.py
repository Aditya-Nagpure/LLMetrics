"""Cost estimation based on token counts and provider pricing tables."""
from llmetrics.core.models import TokenUsage

# USD per 1M tokens — (input_price, output_price)
# Sources: provider pricing pages (approximate, not billed amounts)
PRICING: dict[str, tuple[float, float]] = {
    # Groq (hosted inference)
    "llama3-8b-8192": (0.05, 0.08),
    "llama3-70b-8192": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma-7b-it": (0.07, 0.07),
    # Anthropic Claude
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    # Fallback
    "default": (1.00, 2.00),
}


def estimate_cost(
    tokens: TokenUsage,
    model: str = "default",
    cost_input_per_1m: float | None = None,
    cost_output_per_1m: float | None = None,
) -> float:
    """
    Estimate cost in USD from token counts.
    Override prices take precedence over the PRICING table.
    """
    table_input, table_output = PRICING.get(model, PRICING["default"])
    input_price = cost_input_per_1m if cost_input_per_1m is not None else table_input
    output_price = cost_output_per_1m if cost_output_per_1m is not None else table_output

    return (tokens.prompt * input_price + tokens.completion * output_price) / 1_000_000