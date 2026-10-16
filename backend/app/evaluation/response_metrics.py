"""Deterministic execution-usage, cost, and instruction-check metrics."""

from __future__ import annotations

import json
import re

from app.evaluation.response_config import ResponseEvaluationSettings
from app.evaluation.response_types import (
    CostMetric,
    DeterministicInstructionCheck,
    ResponseEvaluationRequest,
    TokenUsageMetric,
)
from app.semantic.constraints.extractor import BaseConstraintExtractor, RuleConstraintExtractor
from app.semantic.types import Constraint


def calculate_token_usage(request: ResponseEvaluationRequest) -> TokenUsageMetric:
    return TokenUsageMetric(
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=request.resolved_total_tokens(),
    )


def calculate_target_model_cost(
    request: ResponseEvaluationRequest,
    config: ResponseEvaluationSettings,
) -> CostMetric:
    if request.estimated_cost is not None:
        return CostMetric(
            estimated_cost=request.estimated_cost,
            status="provided",
            reason="Estimated execution cost was supplied by the caller.",
        )

    pricing, pricing_error = config.pricing_for(request.target_model)
    if pricing is None:
        reason_code = (
            "INVALID_TARGET_MODEL_PRICING"
            if pricing_error and "configuration is invalid" in pricing_error
            else "TARGET_MODEL_PRICING_MISSING_FOR_MODEL"
            if pricing_error and "No pricing is configured for target model" in pricing_error
            else "TARGET_MODEL_PRICING_NOT_CONFIGURED"
        )
        return CostMetric(
            estimated_cost=None,
            status="unavailable",
            reason_code=reason_code,
            reason=pricing_error or "Target-model pricing is unavailable.",
        )
    if request.input_tokens is None or request.output_tokens is None:
        return CostMetric(
            estimated_cost=None,
            status="unavailable",
            reason_code="TOKEN_USAGE_INCOMPLETE",
            reason="Both input and output token usage are required to estimate execution cost.",
        )

    amount = (
        request.input_tokens * pricing.input_price_per_million
        + request.output_tokens * pricing.output_price_per_million
    ) / 1_000_000
    return CostMetric(
        estimated_cost=amount,
        status="estimated",
        reason_code=None,
        reason="Estimated from configured target-model prices and supplied token usage.",
    )


def _check(
    constraint: Constraint,
    status: str,
    reason: str,
    *,
    expected: str | None = None,
    observed: str | None = None,
) -> DeterministicInstructionCheck:
    return DeterministicInstructionCheck(
        constraint_type=constraint.type,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        expected=expected,
        observed=observed,
    )


def _word_limit_check(constraint: Constraint, response: str) -> DeterministicInstructionCheck | None:
    evidence = (constraint.evidence or "").casefold()
    if constraint.maxWords is None or not re.search(r"\bwords?\b", evidence):
        return None
    words = re.findall(r"\b[\w’'-]+\b", response, flags=re.UNICODE)
    strict_limit = bool(re.search(r"\b(under|less than|fewer than)\b", evidence))
    passed = len(words) < constraint.maxWords if strict_limit else len(words) <= constraint.maxWords
    comparator = "<" if strict_limit else "≤"
    return _check(
        constraint,
        "passed" if passed else "failed",
        "Response word count was compared with the explicit word limit.",
        expected=f"{comparator} {constraint.maxWords} words",
        observed=f"{len(words)} words",
    )


def _numbered_items(response: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for line in response.splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s+(.+?)\s*$", line)
        if match:
            found.append((int(match.group(1)), match.group(2)))
    return found


def _quantity_check(constraint: Constraint, response: str) -> DeterministicInstructionCheck:
    evidence = constraint.evidence or constraint.description
    match = re.search(r"\b(\d+)\s+(examples?|methods?|steps?|items?)\b", evidence, re.I)
    if not match or match.group(2).casefold().rstrip("s") != "example":
        return _check(
            constraint,
            "not_evaluated",
            "This quantity is not a clearly enumerable example list; no reliable automatic count was made.",
        )

    expected = int(match.group(1))
    items = _numbered_items(response)
    numbers = [number for number, _ in items]
    # A numbered list alone could be steps, reasons, or another kind of item.
    # Count it as examples only when the response labels the section/items as examples.
    has_examples_heading = bool(re.search(r"(?im)^\s*examples?\s*:?\s*$", response))
    all_items_labeled_examples = bool(items) and all(
        re.match(r"(?i)^example\b", item_text) for _, item_text in items
    )
    if (
        not items
        or numbers != list(range(1, len(items) + 1))
        or not (has_examples_heading or all_items_labeled_examples)
    ):
        return _check(
            constraint,
            "not_evaluated",
            "A contiguous numbered example list was not detected, so the requested quantity cannot be checked reliably.",
            expected=f"{expected} examples",
        )
    passed = len(items) == expected
    return _check(
        constraint,
        "passed" if passed else "failed",
        "The number of clearly numbered response items was compared with the requested example count.",
        expected=f"{expected} examples",
        observed=f"{len(items)} numbered items",
    )


def _format_check(constraint: Constraint, response: str) -> DeterministicInstructionCheck:
    output_format = constraint.outputFormat
    if output_format == "table":
        lines = response.splitlines()
        has_table = any("|" in line for line in lines) and any(
            re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line)
            for line in lines
        )
        return _check(
            constraint,
            "passed" if has_table else "failed",
            "A Markdown table header and separator row were checked.",
            expected="Markdown table",
            observed="Markdown table detected" if has_table else "No Markdown table detected",
        )
    if output_format == "bullet_list":
        has_bullets = any(re.match(r"^\s*[-*+]\s+\S", line) for line in response.splitlines())
        return _check(
            constraint,
            "passed" if has_bullets else "failed",
            "Bullet markers at the start of response lines were checked.",
            expected="bullet list",
            observed="bullet list detected" if has_bullets else "No bullet list detected",
        )
    if output_format == "numbered_list":
        has_numbering = bool(_numbered_items(response))
        return _check(
            constraint,
            "passed" if has_numbering else "failed",
            "Numbered items at the start of response lines were checked.",
            expected="numbered list",
            observed="numbered list detected" if has_numbering else "No numbered list detected",
        )
    if output_format == "json":
        try:
            json.loads(response)
            valid_json = True
        except (json.JSONDecodeError, TypeError):
            valid_json = False
        return _check(
            constraint,
            "passed" if valid_json else "failed",
            "The response was parsed as a JSON document.",
            expected="valid JSON",
            observed="valid JSON" if valid_json else "invalid JSON",
        )
    return _check(
        constraint,
        "not_evaluated",
        f"The requested output format '{output_format}' has no conservative deterministic checker.",
    )


def calculate_deterministic_instruction_checks(
    original_prompt: str,
    response: str,
    extractor: BaseConstraintExtractor | None = None,
) -> list[DeterministicInstructionCheck]:
    constraints = (extractor or RuleConstraintExtractor()).extract(original_prompt)
    checks: list[DeterministicInstructionCheck] = []
    for constraint in constraints:
        if constraint.type == "length":
            result = _word_limit_check(constraint, response)
            checks.append(result or _check(
                constraint,
                "not_evaluated",
                "Only explicit word limits are checked; this length unit or structure is not measured conservatively.",
            ))
        elif constraint.type == "quantity":
            checks.append(_quantity_check(constraint, response))
        elif constraint.type == "format":
            checks.append(_format_check(constraint, response))
        else:
            checks.append(_check(
                constraint,
                "not_evaluated",
                "This constraint type cannot be evaluated reliably using a deterministic check.",
            ))
    return checks
