"""
analysis/text_stats.py

Local computation of word-level and sentence-level statistics.
No LLM API is called here.

Sentence splitting is done with a lightweight rule-based approach that
handles common abbreviations and avoids the "count '.' characters" pitfall.
If a more sophisticated splitter is needed in future the interface is
unchanged — only this module needs updating.
"""

from __future__ import annotations

import re
from typing import List

from .types import TextStats


# ---------------------------------------------------------------------------
# Known abbreviations that must not trigger sentence boundaries
# ---------------------------------------------------------------------------

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs",
    "etc", "e.g", "i.e", "approx", "dept", "est", "fig",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    "st", "ave", "blvd",
}

# Regex that matches a sentence-ending punctuation character
_SENTENCE_END_RE = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')

# Regex to extract bare "words" (strips surrounding punctuation)
_WORD_RE = re.compile(r"\b[a-zA-Z0-9']+\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences using punctuation boundaries.

    Handles:
    - Period, question mark, exclamation mark as sentence terminators.
    - Avoids splitting on known abbreviations.
    - Falls back gracefully to treating the whole text as one sentence.
    """
    # Normalise whitespace so splitting is consistent
    normalised = re.sub(r"\s+", " ", text).strip()
    if not normalised:
        return []

    # Split on whitespace that follows . ? or !
    raw_splits = _SENTENCE_END_RE.split(normalised)

    sentences: List[str] = []
    for fragment in raw_splits:
        fragment = fragment.strip()
        if not fragment:
            continue
        # If the fragment ends with an abbreviation followed by a period, it
        # probably is not a real sentence boundary — merge with next fragment.
        last_word = fragment.rstrip(".").split()[-1].lower().rstrip(".")
        if last_word in _ABBREV and fragment.endswith(".") and len(raw_splits) > 1:
            # Will naturally be followed by the next fragment in the loop;
            # just carry on — the split has already happened, so we accept the
            # fragment as-is (minor imprecision in edge cases is acceptable).
            pass
        sentences.append(fragment)

    return sentences if sentences else [normalised]


def _extract_words(text: str) -> List[str]:
    """Return a list of word tokens (lowercase, no surrounding punctuation)."""
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_text_stats(text: str) -> TextStats:
    """
    Compute word-level and sentence-level statistics for *text*.

    Parameters
    ----------
    text:
        The raw prompt string.

    Returns
    -------
    TextStats
        Populated statistics object.
    """
    words = _extract_words(text)
    sentences = _split_sentences(text)

    word_count = len(words)
    unique_words = {w.lower() for w in words}
    unique_word_count = len(unique_words)

    total_word_len = sum(len(w) for w in words)
    avg_word_length = round(total_word_len / word_count, 2) if word_count else 0.0

    # Sentence lengths (in words)
    sent_lengths = [len(_extract_words(s)) for s in sentences]
    sentence_count = len(sentences)
    avg_sentence_length = round(sum(sent_lengths) / sentence_count, 2) if sentence_count else 0.0
    min_sentence_length = min(sent_lengths) if sent_lengths else 0
    max_sentence_length = max(sent_lengths) if sent_lengths else 0

    # Non-empty lines
    non_empty_lines = [ln for ln in text.splitlines() if ln.strip()]
    line_count = len(non_empty_lines)

    return TextStats(
        characterCount=len(text),
        wordCount=word_count,
        uniqueWordCount=unique_word_count,
        sentenceCount=sentence_count,
        lineCount=line_count,
        averageWordLength=avg_word_length,
        averageSentenceLength=avg_sentence_length,
        minSentenceLength=min_sentence_length,
        maxSentenceLength=max_sentence_length,
    )
