"""Per-model prices, used to estimate what a run cost.

USD per million tokens: (uncached input, cache read, output). Only models whose prices we
have verified are listed; anything else gets no estimate rather than a wrong one.
Source: Claude API reference, Oct 2026 (https://claude.com/pricing).
"""

PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
    # Haiku 5.5's rate applies to prompts of <=100K tokens; longer prompts cost 5x ($0.50 / $2.50).
    # Agent prompts here stay well under that (largest seen: ~60K), so the low rate is used.
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
}


def estimate_cost(model: str | None, input_tokens: int, cache_read_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost, or None for a model without known prices.

    An estimate: cache *writes* are billed somewhat above normal input, but the trace
    doesn't separate them from other uncached input.
    """
    if model not in PRICES:
        return None
    uncached_price, cache_read_price, output_price = PRICES[model]
    uncached = input_tokens - cache_read_tokens
    return (uncached * uncached_price + cache_read_tokens * cache_read_price + output_tokens * output_price) / 1_000_000
