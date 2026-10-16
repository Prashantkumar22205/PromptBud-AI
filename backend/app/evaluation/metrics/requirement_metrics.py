"""Deterministic structured matching for Module 3 requirements."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.evaluation.types import RequirementPreservationMetric
from app.semantic.requirements.extractor import BaseRequirementExtractor
from app.semantic.types import Requirement


def _normalize(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value or "").casefold()).strip()


_OBJECT_STOP_WORDS = {"a", "an", "the", "how", "works", "me", "through", "with", "in", "of", "to", "for"}


def _object_tokens(value: str | None) -> frozenset[str]:
    tokens = (_normalize(value).split())
    return frozenset(
        "example" if token == "examples" else token
        for token in tokens
        if token not in _OBJECT_STOP_WORDS
    )


def _requirement_matches(original: Requirement, candidate: Requirement) -> bool:
    if original.mandatory != candidate.mandatory:
        return False

    original_action = _normalize(original.action)
    candidate_action = _normalize(candidate.action)
    original_object = _object_tokens(original.object)
    candidate_object = _object_tokens(candidate.object)
    if not original_object or not original_object.issubset(candidate_object):
        return False

    if original_action == candidate_action:
        return True

    # Module 3 can parse "give 3 examples" as a separate provide requirement,
    # but absorb "with 3 examples" into the object of an explain requirement.
    # Treat that explicit, quantity-preserving construction as the same request.
    return _embedded_example_requirement(original, candidate)


def _embedded_example_requirement(original: Requirement, candidate: Requirement) -> bool:
    original_object = _object_tokens(original.object)
    candidate_object = _object_tokens(candidate.object)
    return (
        original.mandatory == candidate.mandatory
        and _normalize(original.action) == "provide"
        and _normalize(candidate.action) == "explain"
        and "example" in original_object
        and any(token.isdigit() for token in original_object)
        and original_object.issubset(candidate_object)
    )


def _preserved_count(original: Sequence[Requirement], optimized: Sequence[Requirement]) -> int:
    remaining = list(enumerate(optimized))
    preserved = 0
    for requirement in original:
        for index, (_, candidate) in enumerate(remaining):
            if _requirement_matches(requirement, candidate):
                preserved += 1
                remaining.pop(index)
                break
        else:
            # The extractor can fold a separate "provide N examples" item into
            # an explain object's "with N examples" phrase. Reuse that already
            # matched compound requirement only for this explicit case.
            if any(_embedded_example_requirement(requirement, candidate) for candidate in optimized):
                preserved += 1
    return preserved


def calculate_requirement_preservation(
    original_prompt: str,
    optimized_prompt: str,
    extractor: BaseRequirementExtractor,
) -> RequirementPreservationMetric:
    original = extractor.extract(original_prompt)
    optimized = extractor.extract(optimized_prompt)
    preserved = _preserved_count(original, optimized)
    total = len(original)
    return RequirementPreservationMetric(
        original_requirements=original,
        optimized_requirements=optimized,
        preserved_count=preserved,
        original_count=total,
        preservation_score=preserved / total if total else 1.0,
    )
