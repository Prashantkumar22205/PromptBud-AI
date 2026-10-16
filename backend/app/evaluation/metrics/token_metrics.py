"""Token count and reduction metrics using the shared tokenizer abstraction."""

from app.analysis.tokenizer import BaseTokenizer
from app.evaluation.types import TokenMetrics


def calculate_token_metrics(
    original_prompt: str,
    optimized_prompt: str,
    tokenizer: BaseTokenizer,
) -> TokenMetrics:
    original_tokens = tokenizer.count_tokens(original_prompt)
    optimized_tokens = tokenizer.count_tokens(optimized_prompt)
    delta = original_tokens - optimized_tokens
    # A percentage reduction is undefined for a zero-token original. Report 0
    # in that edge case rather than emitting NaN/Infinity or dividing by zero.
    reduction = (delta / original_tokens * 100.0) if original_tokens else 0.0
    return TokenMetrics(
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        token_delta=delta,
        token_reduction_percent=reduction,
    )
