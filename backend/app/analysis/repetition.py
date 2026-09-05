"""
analysis/repetition.py

Deterministic, local repetition detection.
No LLM API is called here.

Detects:
  A. Repeated meaningful words (ignoring stopwords)
  B. Repeated 2-gram and 3-gram phrases
  C. A transparent repetition score in [0, 1]

Repetition Score Formula
------------------------
Let  M  = total count of meaningful tokens (stopwords excluded, case-folded,
          punctuation stripped).
Let  RW = Σ (count − 1) for each repeated meaningful word.
Let  RP = Σ (count − 1) for each repeated n-gram (2-word and 3-word).
Then:
    repetitionScore = min(1.0, (RW + RP) / max(1, M))

The score is 0 when nothing repeats.  It approaches 1 when nearly every
meaningful token participates in a repeated pattern.  Values above ~0.3
typically signal noticeable redundancy in a prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Set, Tuple

from .types import RepetitionStats, RepeatedPhrase, RepeatedWord


# ---------------------------------------------------------------------------
# Stopword list
# ---------------------------------------------------------------------------

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "that", "this", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "what", "which", "who", "when", "where", "how", "why",
    "not", "no", "so", "very", "just", "also", "then", "than", "more",
    "into", "about", "up", "out", "all", "each", "any", "some", "other",
    "such", "own", "same",
}

# Regex that extracts bare word tokens (letters, digits, apostrophes)
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z']*\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> List[str]:
    """Return lowercased word tokens (no punctuation)."""
    return [m.group().lower() for m in _TOKEN_RE.finditer(text)]


def _meaningful_tokens(tokens: List[str]) -> List[str]:
    """Filter stopwords from a token list."""
    return [t for t in tokens if t not in _STOPWORDS]


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return all n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_repetition(text: str, min_word_repeat: int = 2, min_phrase_repeat: int = 2) -> RepetitionStats:
    """
    Analyse *text* for word and phrase repetition.

    Parameters
    ----------
    text:
        The raw prompt string.
    min_word_repeat:
        Minimum occurrences for a word to be listed as repeated.
    min_phrase_repeat:
        Minimum occurrences for an n-gram to be listed as repeated.

    Returns
    -------
    RepetitionStats
        Populated repetition statistics including the deterministic score.
    """
    all_tokens = _tokenise(text)
    meaningful = _meaningful_tokens(all_tokens)

    # ── A. Repeated meaningful words ─────────────────────────────────────────
    word_counts = Counter(meaningful)
    repeated_words: List[RepeatedWord] = [
        RepeatedWord(word=w, count=c)
        for w, c in sorted(word_counts.items(), key=lambda kv: -kv[1])
        if c >= min_word_repeat
    ]

    # ── B. Repeated 2-grams and 3-grams ──────────────────────────────────────
    # Use all tokens (including stopwords) for n-grams so that phrases like
    # "easy to understand" are detected naturally.
    repeated_phrases: List[RepeatedPhrase] = []
    for n in (2, 3):
        grams = _ngrams(all_tokens, n)
        gram_counts = Counter(grams)
        for gram, c in sorted(gram_counts.items(), key=lambda kv: -kv[1]):
            if c >= min_phrase_repeat:
                repeated_phrases.append(
                    RepeatedPhrase(phrase=" ".join(gram), count=c, ngramSize=n)
                )

    # Sort phrases: higher count first, then by phrase length
    repeated_phrases.sort(key=lambda p: (-p.count, -p.ngramSize))

    # ── C. Repetition score ───────────────────────────────────────────────────
    M = len(meaningful)
    RW = sum(rw.count - 1 for rw in repeated_words)
    RP = sum(rp.count - 1 for rp in repeated_phrases)
    raw_score = (RW + RP) / max(1, M)
    repetition_score = round(min(1.0, raw_score), 4)

    return RepetitionStats(
        repeatedWords=repeated_words,
        repeatedPhrases=repeated_phrases,
        repetitionScore=repetition_score,
    )
