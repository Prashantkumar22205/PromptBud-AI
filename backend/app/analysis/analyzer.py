"""
analysis/analyzer.py

Central orchestrator that combines all local analysis sub-modules into a
single PromptAnalysis result.

Usage:
    from app.analysis.analyzer import analyze_prompt

    result: PromptAnalysis = analyze_prompt(prompt_text)

No LLM API is called here or in any module it imports.
"""

from __future__ import annotations

import logging

from .text_stats import compute_text_stats
from .tokenizer import get_tokenizer
from .repetition import compute_repetition
from .structure import compute_structure
from .types import PromptAnalysis, TokenStats

logger = logging.getLogger(__name__)


def analyze_prompt(prompt: str) -> PromptAnalysis:
    """
    Run all local analysis modules on *prompt* and return a combined result.

    Pipeline
    --------
    prompt
        ├─► tokenizer        → TokenStats
        ├─► text_stats       → TextStats
        ├─► repetition       → RepetitionStats
        └─► structure        → StructureStats
                                    │
                                    ▼
                             PromptAnalysis

    Parameters
    ----------
    prompt:
        The sanitised prompt text.  Must not be empty.

    Returns
    -------
    PromptAnalysis
        A fully populated analysis result ready to serialise as JSON.

    Notes
    -----
    No LLM API is called at any point in this function or in any module it
    delegates to.  All analysis is deterministic and local.
    """
    logger.info("Starting local prompt analysis (length=%d chars)", len(prompt))

    # ── Token statistics ──────────────────────────────────────────────────────
    tokenizer = get_tokenizer()
    token_count = tokenizer.count_tokens(prompt)
    token_stats = TokenStats(
        tokenCount=token_count,
        tokenizer=tokenizer.name,
    )
    logger.debug("Token stats: count=%d, tokenizer=%s", token_count, tokenizer.name)

    # ── Text / word / sentence statistics ────────────────────────────────────
    text_stats = compute_text_stats(prompt)
    logger.debug("Text stats: words=%d, sentences=%d", text_stats.wordCount, text_stats.sentenceCount)

    # ── Repetition detection ──────────────────────────────────────────────────
    repetition = compute_repetition(prompt)
    logger.debug("Repetition stats: score=%.4f, repeated_words=%d", repetition.repetitionScore, len(repetition.repeatedWords))

    # ── Structural analysis ───────────────────────────────────────────────────
    structure = compute_structure(prompt)
    logger.debug("Structure stats: sections=%s", structure.sections)

    return PromptAnalysis(
        textStats=text_stats,
        tokenStats=token_stats,
        repetition=repetition,
        structure=structure,
    )
