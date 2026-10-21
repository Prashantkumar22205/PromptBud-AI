"""Deterministic structured matching for Module 3 requirements."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.evaluation.metrics.constraint_metrics import preserved_constraint_pairs
from app.evaluation.types import RequirementPreservationMetric
from app.semantic.constraints.extractor import BaseConstraintExtractor
from app.semantic.requirements.extractor import BaseRequirementExtractor
from app.semantic.types import Constraint, Requirement


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


def _requirement_matches(
    original: Requirement,
    candidate: Requirement,
    shared_context_tokens: frozenset[str] = frozenset(),
    preserved_constraints: Sequence[tuple[Constraint, Constraint]] = (),
) -> bool:
    if original.mandatory != candidate.mandatory:
        return False

    original_action, original_object_text = _canonical_action_and_object(original)
    candidate_action, candidate_object_text = _canonical_action_and_object(candidate)
    original_object_text = _remove_preserved_length_clauses(original_object_text, preserved_constraints)
    candidate_object_text = _remove_preserved_length_clauses(candidate_object_text, preserved_constraints)
    original_object = _object_tokens(original_object_text)
    candidate_object = _object_tokens(candidate_object_text)
    missing_object_tokens = original_object - candidate_object
    if (
        not original_object
        or not missing_object_tokens.issubset(shared_context_tokens)
    ):
        return False

    if original_action == candidate_action:
        return True

    # Module 3 can parse "give 3 examples" as a separate provide requirement,
    # but absorb "with 3 examples" into the object of an explain requirement.
    # Treat that explicit, quantity-preserving construction as the same request.
    return _embedded_example_requirement(original, candidate)


def _canonical_action_and_object(requirement: Requirement) -> tuple[str, str]:
    """Normalize a small set of action paraphrases without relaxing object matching."""
    action = _normalize(requirement.action)
    object_text = _normalize(requirement.object)
    object_tokens = object_text.split()

    if action in {"explain", "describe"}:
        return "explain", object_text
    if action == "provide":
        for noun in ("explanation", "description"):
            if noun in object_tokens[:4]:
                # “provide a beginner-friendly explanation of X” expresses explain(X).
                reduced = [token for token in object_tokens if token != noun]
                return "explain", " ".join(reduced)
        if "list" in object_tokens[:3]:
            # “provide a list of X” expresses list(X).
            return "list", " ".join(token for token in object_tokens if token != "list")
    if action in {"give", "provide", "include"}:
        return "provide", object_text
    if action == "list":
        return "list", object_text
    if action in {"compare", "contrast"}:
        return "compare", object_text
    return action, object_text


_LENGTH_CLAUSE_WORDS = {
    "a", "an", "the", "your", "our", "keep", "keeps", "ensure", "ensures",
    "make", "sure", "remain", "remains", "stay", "stays", "explanation",
    "response", "answer", "output", "text", "under", "below", "within",
    "maximum", "max", "limit", "limited", "at", "most", "no", "more",
    "than", "not", "exceed", "exceeds", "word", "words",
}


def _is_preserved_length_clause(
    clause: str,
    preserved_constraints: Sequence[tuple[Constraint, Constraint]],
) -> bool:
    tokens = _normalize(clause).split()
    for original, candidate in preserved_constraints:
        if original.type.casefold() != "length" or original.maxWords is None:
            continue
        if original.maxWords != candidate.maxWords:
            continue
        if str(original.maxWords) not in tokens or not {"word", "words"}.intersection(tokens):
            continue
        has_length_cue = any(
            cue in tokens
            for cue in ("under", "below", "within", "maximum", "max", "limit", "limited", "exceed", "exceeds")
        ) or any(pair in zip(tokens, tokens[1:]) for pair in (("at", "most"), ("no", "more"), ("more", "than"), ("not", "exceed")))
        if has_length_cue and set(tokens).issubset(_LENGTH_CLAUSE_WORDS | {str(original.maxWords)}):
            return True
    return False


def _remove_preserved_length_clauses(
    object_text: str,
    preserved_constraints: Sequence[tuple[Constraint, Constraint]],
) -> str:
    clauses = re.split(r"\band\b", _normalize(object_text))
    retained = [
        clause for clause in clauses
        if not _is_preserved_length_clause(clause, preserved_constraints)
    ]
    # Never erase an entire requirement object as if it were only constraint metadata.
    return " and ".join(retained if retained else clauses)


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


def _preserved_count(
    original: Sequence[Requirement],
    optimized: Sequence[Requirement],
    shared_context_tokens: frozenset[str] = frozenset(),
    preserved_constraints: Sequence[tuple[Constraint, Constraint]] = (),
) -> int:
    remaining = list(enumerate(optimized))
    preserved = 0
    for requirement in original:
        for index, (_, candidate) in enumerate(remaining):
            if _requirement_matches(
                requirement, candidate, shared_context_tokens, preserved_constraints
            ):
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
    constraint_extractor: BaseConstraintExtractor | None = None,
) -> RequirementPreservationMetric:
    original = extractor.extract(original_prompt)
    optimized = extractor.extract(optimized_prompt)
    preserved_constraints = ()
    if constraint_extractor is not None:
        preserved_constraints = preserved_constraint_pairs(
            constraint_extractor.extract(original_prompt),
            constraint_extractor.extract(optimized_prompt),
        )
    shared_context_tokens = frozenset(
        {"c"}
        if re.search(r"(?<!\w)c\+\+(?!\w)", original_prompt, flags=re.IGNORECASE)
        and re.search(r"(?<!\w)c\+\+(?!\w)", optimized_prompt, flags=re.IGNORECASE)
        else set()
    )
    preserved = _preserved_count(original, optimized, shared_context_tokens, preserved_constraints)
    total = len(original)
    return RequirementPreservationMetric(
        original_requirements=original,
        optimized_requirements=optimized,
        preserved_count=preserved,
        original_count=total,
        preservation_score=preserved / total if total else 1.0,
    )
