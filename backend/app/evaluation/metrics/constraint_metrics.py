"""Deterministic structured matching for Module 3 constraints."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.evaluation.types import ConstraintPreservationMetric
from app.semantic.constraints.extractor import BaseConstraintExtractor
from app.semantic.types import Constraint


_VALUE_FIELDS = ("maxWords", "minWords", "outputFormat", "targetLanguage")
_LENGTH_UNITS = {"word", "sentence", "character", "line", "paragraph"}


def _normalize(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value or "").casefold()).strip()


def _explicit_length_unit(constraint: Constraint) -> str | None:
    if _normalize(constraint.type) != "length":
        return None
    words = _normalize(constraint.evidence).split()
    for word in words:
        singular = word[:-1] if word.endswith("s") else word
        if singular in _LENGTH_UNITS:
            return singular
    return None


def _constraint_matches(original: Constraint, candidate: Constraint) -> bool:
    if _normalize(original.type) != _normalize(candidate.type):
        return False

    if _explicit_length_unit(original) != _explicit_length_unit(candidate):
        return False

    # Compare the extractor's typed values whenever the original constraint
    # has them. This protects numeric lengths, formats, and natural languages.
    populated_fields = [field for field in _VALUE_FIELDS if getattr(original, field) is not None]
    if any(getattr(original, field) != getattr(candidate, field) for field in populated_fields):
        return False

    # The current Module 3 schema has no separate typed value for quantity,
    # style, scope, required, or restriction; its normalized description is
    # the structured extractor's canonical representation for those cases.
    if not populated_fields:
        return _normalize(original.description) == _normalize(candidate.description)
    return True


def preserved_constraint_pairs(
    original: Sequence[Constraint], optimized: Sequence[Constraint]
) -> list[tuple[Constraint, Constraint]]:
    remaining = list(optimized)
    preserved: list[tuple[Constraint, Constraint]] = []
    for constraint in original:
        for index, candidate in enumerate(remaining):
            if _constraint_matches(constraint, candidate):
                preserved.append((constraint, candidate))
                remaining.pop(index)
                break
    return preserved


def _preserved_count(original: Sequence[Constraint], optimized: Sequence[Constraint]) -> int:
    return len(preserved_constraint_pairs(original, optimized))


def calculate_constraint_preservation(
    original_prompt: str,
    optimized_prompt: str,
    extractor: BaseConstraintExtractor,
) -> ConstraintPreservationMetric:
    original = extractor.extract(original_prompt)
    optimized = extractor.extract(optimized_prompt)
    preserved = _preserved_count(original, optimized)
    total = len(original)
    return ConstraintPreservationMetric(
        original_constraints=original,
        optimized_constraints=optimized,
        preserved_count=preserved,
        original_count=total,
        preservation_score=preserved / total if total else 1.0,
    )
