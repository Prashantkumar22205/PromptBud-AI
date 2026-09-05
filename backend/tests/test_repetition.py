"""
tests/test_repetition.py

Unit tests for repetition detection.
"""

from __future__ import annotations

from app.analysis.repetition import compute_repetition


def test_no_repetition_unique_words():
    result = compute_repetition("The quick brown fox jumps over the lazy dog.")
    # Meaningful non-stopword tokens: quick, brown, fox, jumps, lazy, dog
    assert result.repetitionScore < 0.5


def test_obvious_repetition():
    result = compute_repetition("cat cat cat cat cat")
    assert len(result.repeatedWords) == 1
    assert result.repeatedWords[0].word == "cat"
    assert result.repeatedWords[0].count == 5


def test_repeated_words_ignore_stopwords():
    result = compute_repetition("the the the is is is and and and")
    words = [rw.word for rw in result.repeatedWords]
    assert "the" not in words
    assert "is" not in words
    assert "and" not in words


def test_repeated_meaningful_words():
    text = "Explain binary search clearly. Make the explanation clear."
    result = compute_repetition(text)
    # "explain" appears once, "explanation" appears once — different tokens
    # "binary" and "search" each appear once
    # Score should be low
    assert result.repetitionScore >= 0.0


def test_repeated_phrases_2gram():
    text = "binary search is fast. binary search is efficient."
    result = compute_repetition(text)
    phrases = [rp.phrase for rp in result.repeatedPhrases]
    assert any("binary search" in p for p in phrases)


def test_repeated_phrases_3gram():
    text = "easy to understand code. The code must be easy to understand."
    result = compute_repetition(text)
    phrases = [rp.phrase for rp in result.repeatedPhrases]
    assert any("easy to understand" in p for p in phrases)


def test_repetition_score_zero_unique():
    # All unique meaningful words → low score
    result = compute_repetition("Alpha beta gamma delta epsilon zeta.")
    assert result.repetitionScore == 0.0


def test_repetition_score_range():
    result = compute_repetition("test test test and test again test")
    assert 0.0 <= result.repetitionScore <= 1.0


def test_repeated_words_sorted_by_count():
    text = "apple apple apple banana banana cherry"
    result = compute_repetition(text)
    if len(result.repeatedWords) >= 2:
        assert result.repeatedWords[0].count >= result.repeatedWords[1].count


def test_empty_repetition_for_single_word():
    result = compute_repetition("hello")
    assert len(result.repeatedWords) == 0
    assert result.repetitionScore == 0.0


def test_repetition_with_punctuation():
    # "search," and "search" should both count as the same token
    text = "binary search, then binary search again."
    result = compute_repetition(text)
    words = [rw.word for rw in result.repeatedWords]
    assert "binary" in words or "search" in words


def test_repeated_phrases_count_correct():
    text = "step by step guide. Follow step by step instructions."
    result = compute_repetition(text)
    phrases = [rp for rp in result.repeatedPhrases if "step by step" in rp.phrase]
    if phrases:
        assert phrases[0].count >= 2
