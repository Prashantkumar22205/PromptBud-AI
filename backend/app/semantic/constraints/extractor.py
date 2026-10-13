"""
semantic/constraints/extractor.py

Constraint Extraction — Module 3 (Non-LLM, Deterministic)

Architecture
------------
Input Prompt
    │
    ▼
spaCy / regex constraint-cue detection
    │
    ▼
Category identification (format, style, length, quantity, language,
                          restriction, scope, required)
    │
    ▼
Value extraction (numeric counts, format tokens, language name, etc.)
    │
    ▼
Span-overlap deduplication (longest evidence span wins)
    │
    ▼
Programming-language guard (prevents C++, Python etc. being emitted
                             as natural-language constraints)
    │
    ▼
Confidence assignment (heuristic, bounded in [0.0, 1.0])
    │
    ▼
Constraint list (Constraint objects with type, description, value fields,
                  confidence, method, evidence)

Design Properties
-----------------
- Non-LLM: zero external AI API calls; entirely deterministic regex/rule logic.
- Reproducible: identical output for identical input on every run.
- Interpretable: every Constraint retains the 'evidence' span and 'method'
  identifier so users and downstream systems can trace the extraction.
- Backward-compatible: existing schema fields (maxWords, outputFormat,
  targetLanguage) are preserved unchanged.

Constraint Types Supported
--------------------------
  length      — word/character count limits, brevity adverbs ("briefly")
  format      — output structure ("table", "bullet list", "JSON", "step by step")
  style       — register / voice ("simple language", "formal tone", "briefly")
  quantity    — numeric count ("5 examples", "three methods")
  language    — natural output language ("in Hindi", "respond in French")
  restriction — excluded techniques / tools ("do not use recursion", "without LLM")
  scope       — topic boundaries ("only discuss X")
  required    — mandatory inclusions ("must include", "make sure")

NOT extracted as constraints
----------------------------
  - Programming languages (C++, Python, Java …) — handled by
    context.programmingLanguage (LANGUAGE_PATTERNS in patterns.py)
  - Audience ("for a beginner") — handled by context.audience
  - Topic ("explain binary search") — handled by requirements + context.topic

Limitations
-----------
  - Implicit preferences ("make it nice") cannot be detected without an LLM.
  - Long-range dependencies ("as I mentioned … in table form") may be missed.
  - Unusual phrasing ("give it a step-by-step feel") may not match any pattern.
  - Ambiguous evidence ("in Python" is a programming language, not a style) is
    guarded by PROGRAMMING_LANGUAGE_NAMES but edge-cases may exist.

No LLM API is called.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Word-number → integer mapping for quantity extraction
WORD_NUMBERS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Extended natural language list for targetLanguage extraction
NATURAL_LANGUAGE_PATTERN = re.compile(
    r"\b(hindi|french|german|spanish|chinese|japanese|arabic|portuguese|"
    r"russian|italian|korean|dutch|turkish|polish|swedish|danish|norwegian|"
    r"finnish|greek|hebrew|thai|vietnamese|malay|indonesian|bengali|urdu|"
    r"persian|farsi|english)\b",
    re.I,
)

# Format token → outputFormat value mapping
FORMAT_TOKEN_MAP: Dict[str, str] = {
    "bullet": "bullet_list",
    "bulleted": "bullet_list",
    "numbered": "numbered_list",
    "ordered": "numbered_list",
    "table": "table",
    "tabular": "table",
    "json": "json",
    "xml": "xml",
    "csv": "csv",
    "markdown": "markdown",
    "html": "html",
    "yaml": "yaml",
    "chart": "chart",
    "diagram": "diagram",
    "outline": "outline",
    "report": "report",
    "list": "list",
    "step": "step_by_step",
    "step-by-step": "step_by_step",
    "step by step": "step_by_step",
    "headings": "headings",
    "heading": "headings",
    "code only": "code_only",
}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseConstraintExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, prompt: str) -> list:
        """Return a list of Constraint objects."""


# ---------------------------------------------------------------------------
# Improved RuleConstraintExtractor implementation
# ---------------------------------------------------------------------------


class RuleConstraintExtractor(BaseConstraintExtractor):
    """
    Constraint extractor using configurable regex patterns from utils/patterns.py.

    To add a new constraint type or cue pattern, update CONSTRAINT_PATTERNS in
    utils/patterns.py only.  The extractor logic is generic.
    """

    def extract(self, prompt: str) -> list:
        from app.semantic.types import Constraint
        from app.semantic.utils.patterns import CONSTRAINT_PATTERNS, PROGRAMMING_LANGUAGE_NAMES

        results: List[Constraint] = []
        # Map from normalised evidence → (start, end, Constraint) for overlap dedup
        seen_spans: List[Tuple[int, int, str]] = []  # (start, end, norm_evidence)

        def _is_subsumed(start: int, end: int) -> bool:
            """Return True if [start, end) is fully contained in an already-seen span."""
            for s, e, _ in seen_spans:
                if s <= start and end <= e:
                    return True
            return False

        for c_type, pattern, base_confidence in CONSTRAINT_PATTERNS:
            for m in pattern.finditer(prompt):
                evidence = m.group(0).strip()
                span_start = m.start()
                span_end = m.end()

                # --- Guard: do NOT emit programming language names as type=language ---
                if c_type == "language":
                    lang_lower = evidence.lower()
                    # Check if any programming language name is the dominant content
                    for prog_lang in PROGRAMMING_LANGUAGE_NAMES:
                        if re.search(r"\b" + re.escape(prog_lang) + r"\b", lang_lower):
                            # Check that it's NOT also a natural language match
                            nat_match = NATURAL_LANGUAGE_PATTERN.search(evidence)
                            if not nat_match:
                                logger.debug("Skipping programming language as language constraint: %s", evidence)
                                break  # skip this match
                    else:
                        pass  # natural language match OK — fall through to extraction
                    # Verify there's a natural language name in the evidence
                    if not NATURAL_LANGUAGE_PATTERN.search(evidence):
                        continue

                # --- Span-overlap deduplication (longest wins) ---
                # Normalise: strip trailing punctuation so "Do not use recursion"
                # and "Do not use recursion." are treated as the same evidence.
                norm = re.sub(r"[.,;:!?]+$", "", evidence.strip())
                norm = re.sub(r"\s+", " ", norm.lower())

                # Check if this span is fully inside an already-seen larger span
                if _is_subsumed(span_start, span_end):
                    continue

                # Also skip if the normalised evidence text is already represented
                if any(n == norm for _, _, n in seen_spans):
                    continue

                # Remove any previously seen spans that are fully contained in
                # this new (larger) match — larger match wins
                seen_spans = [
                    (s, e, n) for s, e, n in seen_spans
                    if not (span_start <= s and e <= span_end)
                ]
                seen_spans.append((span_start, span_end, norm))

                # --- Field extraction ---
                max_words: Optional[int] = None
                min_words: Optional[int] = None
                output_fmt: Optional[str] = None
                target_lang: Optional[str] = None
                quantity_value: Optional[str] = None

                if c_type == "length":
                    digit_match = re.search(r"\d+", evidence)
                    if digit_match:
                        max_words = int(digit_match.group(0))

                elif c_type == "format":
                    ev_lower = evidence.lower()
                    # Try most-specific token matches first
                    if "step-by-step" in ev_lower or "step by step" in ev_lower:
                        output_fmt = "step_by_step"
                    elif "bullet" in ev_lower:
                        output_fmt = "bullet_list"
                    elif "numbered" in ev_lower or "ordered" in ev_lower:
                        output_fmt = "numbered_list"
                    elif "tabular" in ev_lower or "table" in ev_lower:
                        output_fmt = "table"
                    elif "json" in ev_lower:
                        output_fmt = "json"
                    elif "xml" in ev_lower:
                        output_fmt = "xml"
                    elif "csv" in ev_lower:
                        output_fmt = "csv"
                    elif "markdown" in ev_lower:
                        output_fmt = "markdown"
                    elif "html" in ev_lower:
                        output_fmt = "html"
                    elif "yaml" in ev_lower:
                        output_fmt = "yaml"
                    elif "heading" in ev_lower:
                        output_fmt = "headings"
                    elif "code only" in ev_lower:
                        output_fmt = "code_only"
                    elif "report" in ev_lower:
                        output_fmt = "report"
                    elif "outline" in ev_lower:
                        output_fmt = "outline"
                    elif "chart" in ev_lower:
                        output_fmt = "chart"
                    elif "diagram" in ev_lower:
                        output_fmt = "diagram"
                    elif "list" in ev_lower:
                        output_fmt = "list"

                elif c_type == "language":
                    nat_match = NATURAL_LANGUAGE_PATTERN.search(evidence)
                    if nat_match:
                        target_lang = nat_match.group(0).capitalize()

                elif c_type == "quantity":
                    # Extract numeric or word-number value
                    num_match = re.search(r"\d+", evidence)
                    if num_match:
                        quantity_value = num_match.group(0)
                    else:
                        word_match = re.search(
                            r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
                            evidence, re.I
                        )
                        if word_match:
                            quantity_value = str(WORD_NUMBERS.get(word_match.group(0).lower(), word_match.group(0)))

                # --- Build description ---
                description = self._build_description(c_type, evidence, quantity_value, output_fmt, target_lang, max_words)

                results.append(Constraint(
                    type=c_type,
                    description=description,
                    maxWords=max_words,
                    minWords=min_words,
                    outputFormat=output_fmt,
                    targetLanguage=target_lang,
                    confidence=round(base_confidence, 2),
                    method="rule",
                    evidence=evidence,
                ))

        return results

    def _build_description(
        self,
        c_type: str,
        evidence: str,
        quantity_value: Optional[str],
        output_fmt: Optional[str],
        target_lang: Optional[str],
        max_words: Optional[int],
    ) -> str:
        """Build a human-readable description for each constraint type."""
        if c_type == "length":
            if max_words:
                return f"Length limit: maximum {max_words} words"
            return f"Length constraint: {evidence}"
        if c_type == "format":
            if output_fmt:
                return f"Output format: {output_fmt.replace('_', ' ')}"
            return f"Format constraint: {evidence}"
        if c_type == "style":
            return f"Style constraint: {evidence}"
        if c_type == "quantity":
            if quantity_value:
                return f"Quantity constraint: {quantity_value} items requested"
            return f"Quantity constraint: {evidence}"
        if c_type == "language":
            if target_lang:
                return f"Output language: {target_lang}"
            return f"Language constraint: {evidence}"
        if c_type in ("restriction", "forbidden"):
            return f"Restriction: {evidence}"
        if c_type == "scope":
            return f"Scope constraint: {evidence}"
        if c_type == "required":
            return f"Required element: {evidence}"
        return f"Constraint ({c_type}): {evidence}"


def get_constraint_extractor() -> BaseConstraintExtractor:
    return RuleConstraintExtractor()

