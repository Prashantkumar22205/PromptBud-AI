"""Ambiguity detection for prompts.

Phase 3A: Rule-based detector for conflicting constraints, underspecified objects,
and missing critical context.
Phase 3B: Can be extended/replaced with NLI cross-encoder models for contradiction detection.
"""

from abc import ABC, abstractmethod
import re
from typing import List, Optional
from app.semantic.types import Ambiguity, Constraint, ContextResult
from app.semantic.utils.patterns import (
    CONFLICTING_CONSTRAINT_PAIRS,
    UNDERSPECIFIED_PATTERNS,
)


class BaseAmbiguityDetector(ABC):
    """Abstract base class for ambiguity detectors."""

    @abstractmethod
    def detect(
        self,
        prompt: str,
        constraints: Optional[List[Constraint]] = None,
        context: Optional[ContextResult] = None,
    ) -> List[Ambiguity]:
        """Detect ambiguities, conflicts, and missing information in prompt.

        Args:
            prompt: Original prompt string.
            constraints: Previously extracted constraints.
            context: Previously extracted context.

        Returns:
            List of Ambiguity objects.
        """
        pass


class RuleAmbiguityDetector(BaseAmbiguityDetector):
    """Rule-based implementation of ambiguity detector for Phase 3A."""

    def detect(
        self,
        prompt: str,
        constraints: Optional[List[Constraint]] = None,
        context: Optional[ContextResult] = None,
    ) -> List[Ambiguity]:
        """Detect ambiguities in prompt.

        Args:
            prompt: Original prompt text.
            constraints: Optional extracted constraints.
            context: Optional extracted context.

        Returns:
            List of Ambiguity objects.
        """
        ambiguities: List[Ambiguity] = []

        # 1. Detect conflicting constraints
        conflicts = self._detect_conflicting_constraints(prompt, constraints)
        ambiguities.extend(conflicts)

        # 2. Detect underspecified objects / vague phrasing
        underspecified = self._detect_underspecified(prompt)
        ambiguities.extend(underspecified)

        # 3. Detect missing context
        missing_ctx = self._detect_missing_context(prompt, context)
        ambiguities.extend(missing_ctx)

        return ambiguities

    def _detect_conflicting_constraints(
        self,
        prompt: str,
        constraints: Optional[List[Constraint]] = None,
    ) -> List[Ambiguity]:
        """Check for conflicting instructions or constraints."""
        conflicts: List[Ambiguity] = []
        prompt_lower = prompt.lower()

        # Check raw prompt against conflicting keyword pairs
        for label_a, label_b, pattern_a, pattern_b in CONFLICTING_CONSTRAINT_PAIRS:
            match_a = pattern_a.search(prompt)
            match_b = pattern_b.search(prompt)
            if match_a and match_b:
                conflicts.append(
                    Ambiguity(
                        type="conflicting_constraints",
                        description=f"Prompt contains conflicting instructions: '{label_a}' vs '{label_b}'.",
                        severity="high",
                        evidence=f"'{match_a.group(0)}' vs '{match_b.group(0)}'",
                        suggestedResolution=f"Clarify whether the response should follow '{label_a}' or '{label_b}'.",
                    )
                )


        # Also check extracted constraint list types
        if constraints and len(constraints) >= 2:
            types_present = [c.type for c in constraints]
            # Check length constraint conflict (e.g., brief vs detailed)
            brief_found = any("brief" in c.description.lower() or "short" in c.description.lower() for c in constraints)
            detailed_found = any("detail" in c.description.lower() or "comprehensive" in c.description.lower() for c in constraints)
            if brief_found and detailed_found:
                # Avoid duplicate if already caught by keyword pair check
                if not any(a.type == "conflicting_constraints" for a in conflicts):
                    conflicts.append(
                        Ambiguity(
                            type="conflicting_constraints",
                            description="Constraints require both brevity and exhaustive detail simultaneously.",
                            severity="high",
                            evidence="brief/short constraint combined with detailed/comprehensive constraint",
                            suggestedResolution="Specify desired length range or depth level explicitly.",
                        )
                    )

        return conflicts

    def _detect_underspecified(self, prompt: str) -> List[Ambiguity]:
        """Detect vague terms and underspecified objectives."""
        results: List[Ambiguity] = []
        for category, pattern, resolution in UNDERSPECIFIED_PATTERNS:
            match = pattern.search(prompt)
            if match:
                results.append(
                    Ambiguity(
                        type=category,
                        description="Vague or subjective phrasing detected.",
                        severity="medium",
                        evidence=match.group(0),
                        suggestedResolution=resolution,
                    )
                )
        return results


    def _detect_missing_context(
        self,
        prompt: str,
        context: Optional[ContextResult] = None,
    ) -> List[Ambiguity]:
        """Detect missing critical context or target specifications."""
        results: List[Ambiguity] = []
        words = prompt.strip().split()

        # Extremely short prompt without clear domain/context
        domain_val = context.domain.value if (context and context.domain) else None
        if len(words) <= 3 and (domain_val is None or domain_val == "general"):
            results.append(
                Ambiguity(
                    type="missing_context",
                    description="Prompt is very brief and lacks domain/situational context.",
                    severity="low",
                    evidence=prompt.strip(),
                    suggestedResolution="Add specific target domain, goal details, or target audience.",
                )
            )

        # Code generation request without language specification
        code_intent_triggers = ["write code", "code for", "implement a function", "script to", "create an app"]
        prompt_lower = prompt.lower()
        if any(trigger in prompt_lower for trigger in code_intent_triggers):
            lang_val = context.programmingLanguage.value if (context and context.programmingLanguage) else None
            if not lang_val:
                results.append(
                    Ambiguity(
                        type="missing_context",
                        description="Code implementation requested but programming language is not specified.",
                        severity="medium",
                        evidence="code/implementation request without explicit language",
                        suggestedResolution="Specify preferred programming language (e.g., Python, TypeScript).",
                    )
                )

        return results

