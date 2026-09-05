"""
analysis/structure.py

Lightweight structural signal detection.
No LLM API is called here.

IMPORTANT LIMITATION:
    This module uses regex patterns and keyword matching to detect *structural
    signals* in a prompt.  It does NOT perform deep semantic understanding.
    Detections are heuristic; they may produce false positives or miss
    unconventional phrasings.  This is intentional for the baseline module.

Detected signals:
    hasRoleInstruction  — "you are …" / "act as …" patterns
    hasContext          — audience / background context cues
    hasTaskInstruction  — action-verb instruction (explain, write, create, …)
    hasOutputFormat     — output-format directive (bullet points, JSON, …)
    hasConstraints      — explicit constraint (do not …, keep it …, limit …)
    hasExamples         — example marker (for example, e.g., such as, …)
    hasQuestion         — interrogative sentence ending with "?"
"""

from __future__ import annotations

import re
from typing import List

from .types import StructureStats


# ---------------------------------------------------------------------------
# Pattern groups
# ---------------------------------------------------------------------------

# Role instruction: "you are a …", "act as a …", "you're an …", etc.
_ROLE_PATTERNS = [
    re.compile(r"\byou\s+are\s+(a|an|the)?\s*\w", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(a|an|the)?\s*\w", re.IGNORECASE),
    re.compile(r"\byou're\s+(a|an|the)?\s*\w", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\s+(a|an|the)?\s*\w", re.IGNORECASE),
    re.compile(r"\bbehave\s+as\s+(a|an|the)?\s*\w", re.IGNORECASE),
    re.compile(r"\btake\s+the\s+role\s+of\b", re.IGNORECASE),
    re.compile(r"\bassume\s+the\s+role\s+of\b", re.IGNORECASE),
]

# Context / audience signals
_CONTEXT_PATTERNS = [
    re.compile(r"\bfor\s+(a\s+)?beginner", re.IGNORECASE),
    re.compile(r"\bfor\s+student", re.IGNORECASE),
    re.compile(r"\bfor\s+(a\s+)?(junior|senior|expert|novice|advanced)\b", re.IGNORECASE),
    re.compile(r"\bto\s+(a\s+)?(junior|senior|expert|novice|advanced|beginner)\b", re.IGNORECASE),
    re.compile(r"\btarget\s+audience\b", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+for\b", re.IGNORECASE),
    re.compile(r"\bassume\s+that\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(knows|understands|has)\b", re.IGNORECASE),
    re.compile(r"\bwho\s+(is|are)\b", re.IGNORECASE),
    re.compile(r"\bbackground\s+(is|in)\b", re.IGNORECASE),
    re.compile(r"\bcontext\s*:", re.IGNORECASE),
]

# Task / instruction action verbs
_TASK_VERBS = (
    r"explain|write|generate|create|summarize|summarise|compare|analyse|analyze"
    r"|calculate|translate|describe|implement|solve|list|identify|design|build"
    r"|develop|outline|define|evaluate|provide|give|show|tell|teach|demonstrate"
    r"|review|assess|suggest|recommend|find|produce|draft|convert|extract"
)
_TASK_PATTERNS = [
    re.compile(r"\b(?:" + _TASK_VERBS + r")\b", re.IGNORECASE),
]

# Output format directives
_FORMAT_PATTERNS = [
    re.compile(r"\bbullet\s+point", re.IGNORECASE),
    re.compile(r"\bnumbered\s+(list|step)", re.IGNORECASE),
    re.compile(r"\buse\s+heading", re.IGNORECASE),
    re.compile(r"\bformat\s+(as|it|the)", re.IGNORECASE),
    re.compile(r"\breturn\s+json\b", re.IGNORECASE),
    re.compile(r"\bprovide\s+a\s+table", re.IGNORECASE),
    re.compile(r"\bgive\s+(a\s+)?table", re.IGNORECASE),
    re.compile(r"\buse\s+a\s+table", re.IGNORECASE),
    re.compile(r"\bin\s+(json|xml|csv|markdown|html)\s+format", re.IGNORECASE),
    re.compile(r"\bformat\s*:", re.IGNORECASE),
    re.compile(r"\boutput\s*:", re.IGNORECASE),
    re.compile(r"\bresponse\s+format\b", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+following\s+format\b", re.IGNORECASE),
    re.compile(r"\buse\s+headings\b", re.IGNORECASE),
    re.compile(r"\buse\s+subheadings\b", re.IGNORECASE),
    re.compile(r"\bwith\s+headings\b", re.IGNORECASE),
]

# Constraint markers
_CONSTRAINT_PATTERNS = [
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bdon'?t\b", re.IGNORECASE),
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(use|include|repeat|exceed|add)\b", re.IGNORECASE),
    re.compile(r"\bnever\s+(use|include|add|repeat)\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(it|the\s+answer)\s+(short|brief|concise|simple)\b", re.IGNORECASE),
    re.compile(r"\blimit\s+(the|your|it)\b", re.IGNORECASE),
    re.compile(r"\bmust\s+(include|not|be)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+exceed\b", re.IGNORECASE),
    re.compile(r"\bno\s+more\s+than\b", re.IGNORECASE),
    re.compile(r"\buse\s+simple\s+language\b", re.IGNORECASE),
    re.compile(r"\buse\s+plain\s+language\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+using\b", re.IGNORECASE),
    re.compile(r"\bstay\s+within\b", re.IGNORECASE),
    re.compile(r"\bmake\s+sure\b", re.IGNORECASE),
    re.compile(r"\bensure\s+that\b", re.IGNORECASE),
]

# Example markers
_EXAMPLE_PATTERNS = [
    re.compile(r"\bfor\s+example\b", re.IGNORECASE),
    re.compile(r"e\.g\.", re.IGNORECASE),            # No word boundary — period is not \w
    re.compile(r"i\.e\.", re.IGNORECASE),
    re.compile(r"\bsuch\s+as\b", re.IGNORECASE),
    re.compile(r"\bexample\b", re.IGNORECASE),        # Catches 'give an example', 'code example'
    re.compile(r"\bconsider\s+the\s+following", re.IGNORECASE),
    re.compile(r"\bhere\s+is\s+an?\b", re.IGNORECASE),
    re.compile(r"\bfor\s+instance\b", re.IGNORECASE),
    re.compile(r"```"),                                # code block
    re.compile(r"\blike\s+this\b", re.IGNORECASE),
    re.compile(r"\bdry\s+run\b", re.IGNORECASE),      # 'Perform a dry run'
    re.compile(r"\bsample\b", re.IGNORECASE),
    re.compile(r"\bdemonstrate\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _has_question_sentence(text: str) -> bool:
    """Return True if any sentence ends with a '?' character."""
    return bool(re.search(r"\?", text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_structure(text: str) -> StructureStats:
    """
    Detect structural signals in *text* using rule-based pattern matching.

    Parameters
    ----------
    text:
        The raw prompt string.

    Returns
    -------
    StructureStats
        Populated structural detection results.

    Notes
    -----
    This function performs *structural* detection only.  It does NOT claim
    to fully understand the semantic intent of the prompt.
    """
    has_role = _matches_any(text, _ROLE_PATTERNS)
    has_context = _matches_any(text, _CONTEXT_PATTERNS)
    has_task = _matches_any(text, _TASK_PATTERNS)
    has_format = _matches_any(text, _FORMAT_PATTERNS)
    has_constraints = _matches_any(text, _CONSTRAINT_PATTERNS)
    has_examples = _matches_any(text, _EXAMPLE_PATTERNS)
    has_question = _has_question_sentence(text)

    # Build human-readable sections list for the response
    sections: List[str] = []
    if has_role:
        sections.append("Role Instruction")
    if has_context:
        sections.append("Context / Audience")
    if has_task:
        sections.append("Task Instruction")
    if has_format:
        sections.append("Output Format")
    if has_constraints:
        sections.append("Constraints")
    if has_examples:
        sections.append("Examples")
    if has_question:
        sections.append("Question")

    return StructureStats(
        hasRoleInstruction=has_role,
        hasContext=has_context,
        hasTaskInstruction=has_task,
        hasOutputFormat=has_format,
        hasConstraints=has_constraints,
        hasExamples=has_examples,
        hasQuestion=has_question,
        sections=sections,
    )
