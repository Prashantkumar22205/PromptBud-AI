"""
semantic/constraints/extractor.py

Constraint extraction from prompts.

Detects:
  - length    ("keep under 200 words")
  - format    ("use bullet points", "in JSON")
  - style     ("use simple language", "formal tone")
  - language  ("answer in Hindi")
  - technology ("use Python")
  - forbidden ("do not use external libraries")
  - required  ("must include", "make sure")
  - scope     ("only discuss binary search")
  - audience  ("aimed at non-technical readers")

No LLM API is called.

Phase 3B extension
------------------
BaseConstraintExtractor is defined here for future NLI-based detection.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseConstraintExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, prompt: str) -> list:
        """Return a list of Constraint objects."""


# ---------------------------------------------------------------------------
# Phase 3A implementation
# ---------------------------------------------------------------------------


class RuleConstraintExtractor(BaseConstraintExtractor):
    """
    Constraint extractor using configurable regex patterns.

    All patterns are defined in utils/patterns.py (CONSTRAINT_PATTERNS).
    To add a new constraint type or pattern, update that file only.
    """

    def extract(self, prompt: str) -> list:
        from app.semantic.types import Constraint
        from app.semantic.utils.patterns import CONSTRAINT_PATTERNS

        results: List[Constraint] = []
        seen_evidence: Set[str] = set()

        for c_type, pattern, base_confidence in CONSTRAINT_PATTERNS:
            for m in pattern.finditer(prompt):
                evidence = m.group(0).strip()
                # Deduplicate on normalised evidence text
                norm = re.sub(r"\s+", " ", evidence.lower())
                if norm in seen_evidence:
                    continue
                seen_evidence.add(norm)

                max_words = None
                output_fmt = None
                target_lang = None

                if c_type == "length":
                    digit_match = re.search(r"\d+", evidence)
                    if digit_match:
                        max_words = int(digit_match.group(0))

                elif c_type == "format":
                    ev_lower = evidence.lower()
                    if "bullet" in ev_lower:
                        output_fmt = "bullet_list"
                    elif "json" in ev_lower:
                        output_fmt = "json"
                    elif "table" in ev_lower:
                        output_fmt = "table"
                    elif "markdown" in ev_lower:
                        output_fmt = "markdown"

                elif c_type == "language":
                    lang_match = re.search(r"\b(french|spanish|hindi|german|chinese|japanese|russian|english|italian|portuguese)\b", evidence, re.I)
                    if lang_match:
                        target_lang = lang_match.group(0).capitalize()

                results.append(Constraint(
                    type=c_type,
                    description=f"Constraint on {c_type}: {evidence}",
                    maxWords=max_words,
                    outputFormat=output_fmt,
                    targetLanguage=target_lang,
                    confidence=round(base_confidence, 2),
                    method="rule",
                    evidence=evidence,
                ))

        return results



def get_constraint_extractor() -> BaseConstraintExtractor:
    return RuleConstraintExtractor()
