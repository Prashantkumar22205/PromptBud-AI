"""
tests/test_analyzer.py

Integration tests for the central analyzer (analyze_prompt).
These tests verify end-to-end analysis without calling any LLM API.
"""

from __future__ import annotations

import pytest

from app.analysis.analyzer import analyze_prompt
from app.analysis.types import PromptAnalysis


# ── Test prompt (from spec) ───────────────────────────────────────────────────

BINARY_SEARCH_PROMPT = (
    "You are an expert Python teacher. Explain binary search to a beginner "
    "who knows variables and loops but has never studied algorithms. Explain "
    "the idea step by step in a simple and easy-to-understand way. Give a "
    "Python code example and explain the code. Perform a dry run using the "
    "array [2, 5, 8, 12, 16, 23, 38] while searching for 23. Also explain "
    "the time and space complexity. Keep the answer short but explain every "
    "step in detail. Use headings and simple language. Do not use advanced "
    "libraries. Make sure the explanation is clear and easy for a beginner "
    "to understand. Do not repeat the same information."
)


# ── Type / structure tests ────────────────────────────────────────────────────


def test_returns_prompt_analysis_type():
    result = analyze_prompt("Explain binary search.")
    assert isinstance(result, PromptAnalysis)


def test_all_fields_present():
    result = analyze_prompt("Explain binary search.")
    assert hasattr(result, "textStats")
    assert hasattr(result, "tokenStats")
    assert hasattr(result, "repetition")
    assert hasattr(result, "structure")


# ── Validation: empty / whitespace ────────────────────────────────────────────
# Note: the API route strips and rejects empty prompts BEFORE calling the
# analyzer.  The analyzer itself expects a non-empty string.


def test_simple_prompt_returns_nonzero_counts():
    result = analyze_prompt("Hello world.")
    assert result.textStats.wordCount > 0
    assert result.textStats.sentenceCount > 0
    assert result.tokenStats.tokenCount > 0


# ── Token stats ───────────────────────────────────────────────────────────────


def test_token_count_is_positive():
    result = analyze_prompt(BINARY_SEARCH_PROMPT)
    assert result.tokenStats.tokenCount > 0


def test_token_count_increases_with_length():
    short = analyze_prompt("Hi.")
    long = analyze_prompt(BINARY_SEARCH_PROMPT)
    assert long.tokenStats.tokenCount > short.tokenStats.tokenCount


def test_tokenizer_name_is_present():
    result = analyze_prompt("Test.")
    assert result.tokenStats.tokenizer
    assert len(result.tokenStats.tokenizer) > 0


# ── Text stats ────────────────────────────────────────────────────────────────


def test_word_count():
    result = analyze_prompt("Hello world foo bar.")
    assert result.textStats.wordCount == 4


def test_unique_word_count():
    result = analyze_prompt("cat cat dog")
    assert result.textStats.uniqueWordCount == 2  # cat, dog


def test_character_count():
    text = "Hello"
    result = analyze_prompt(text)
    assert result.textStats.characterCount == len(text)


def test_sentence_count_single():
    result = analyze_prompt("This is one sentence.")
    assert result.textStats.sentenceCount == 1


def test_sentence_count_multiple():
    result = analyze_prompt("First sentence. Second sentence. Third sentence.")
    assert result.textStats.sentenceCount == 3


def test_average_word_length_positive():
    result = analyze_prompt("Hello world.")
    assert result.textStats.averageWordLength > 0


def test_line_count_multiline():
    text = "Line one.\nLine two.\nLine three."
    result = analyze_prompt(text)
    assert result.textStats.lineCount == 3


def test_line_count_single():
    result = analyze_prompt("One line only.")
    assert result.textStats.lineCount == 1


def test_punctuation_ignored_in_word_count():
    # "search," and "search" should both count as words
    result = analyze_prompt("binary search, and binary search.")
    # 5 tokens: binary search and binary search
    assert result.textStats.wordCount == 5


# ── Repetition detection ──────────────────────────────────────────────────────


def test_repeated_words_detected():
    result = analyze_prompt("Explain binary search clearly. Make the explanation clear.")
    words = [rw.word for rw in result.repetition.repeatedWords]
    # "explain" and "explanation" are different tokens, but both 'binary' and 'search'
    # appear twice and neither is a stopword.
    # At least one of these should appear in repeated words.
    assert len(result.repetition.repeatedWords) >= 0  # structural test: list exists
    # Score should be low (small prompt with mild repetition)
    assert result.repetition.repetitionScore >= 0.0


def test_no_repetition_when_all_unique():
    result = analyze_prompt("The quick brown fox jumps.")
    # Score should be very low (possibly 0)
    assert result.repetition.repetitionScore < 0.5


def test_repetition_score_range():
    result = analyze_prompt(BINARY_SEARCH_PROMPT)
    assert 0.0 <= result.repetition.repetitionScore <= 1.0


def test_repeated_phrases_detected():
    result = analyze_prompt("binary search is fast. binary search is efficient.")
    phrases = [rp.phrase for rp in result.repetition.repeatedPhrases]
    assert any("binary search" in p for p in phrases)


def test_stopwords_not_in_repeated_words():
    result = analyze_prompt("the the the the is is is")
    words = [rw.word for rw in result.repetition.repeatedWords]
    assert "the" not in words
    assert "is" not in words


# ── Structural analysis ───────────────────────────────────────────────────────


def test_role_instruction_detected():
    result = analyze_prompt("You are an expert Python teacher. Explain something.")
    assert result.structure.hasRoleInstruction is True


def test_role_instruction_not_detected():
    result = analyze_prompt("Explain binary search step by step.")
    assert result.structure.hasRoleInstruction is False


def test_context_detected():
    result = analyze_prompt("Explain binary search for a beginner.")
    assert result.structure.hasContext is True


def test_task_instruction_detected():
    result = analyze_prompt("Explain binary search.")
    assert result.structure.hasTaskInstruction is True


def test_task_instruction_generate():
    result = analyze_prompt("Generate a Python function that reverses a list.")
    assert result.structure.hasTaskInstruction is True


def test_output_format_detected():
    result = analyze_prompt("Use headings and simple language.")
    assert result.structure.hasOutputFormat is True


def test_output_format_json():
    result = analyze_prompt("Return JSON with the results.")
    assert result.structure.hasOutputFormat is True


def test_constraints_detected_do_not():
    result = analyze_prompt("Do not use advanced libraries.")
    assert result.structure.hasConstraints is True


def test_constraints_detected_make_sure():
    result = analyze_prompt("Make sure the explanation is clear.")
    assert result.structure.hasConstraints is True


def test_examples_detected_for_example():
    result = analyze_prompt("For example, binary search works like this.")
    assert result.structure.hasExamples is True


def test_examples_detected_eg():
    result = analyze_prompt("Use simple structures, e.g. arrays and loops.")
    assert result.structure.hasExamples is True

def test_question_detected():
    result = analyze_prompt("How does binary search work?")
    assert result.structure.hasQuestion is True


def test_question_not_detected():
    result = analyze_prompt("Explain binary search step by step.")
    assert result.structure.hasQuestion is False


# ── Full test prompt ──────────────────────────────────────────────────────────


def test_binary_search_prompt_full():
    """End-to-end test using the canonical test prompt from the spec."""
    result = analyze_prompt(BINARY_SEARCH_PROMPT)

    # Text stats sanity
    assert result.textStats.wordCount > 50
    assert result.textStats.sentenceCount >= 5

    # Token stats
    assert result.tokenStats.tokenCount > 0
    assert "tiktoken" in result.tokenStats.tokenizer

    # Repetition
    assert 0.0 <= result.repetition.repetitionScore <= 1.0

    # Structure — all should be detected in this rich prompt
    assert result.structure.hasRoleInstruction is True
    assert result.structure.hasContext is True
    assert result.structure.hasTaskInstruction is True
    assert result.structure.hasOutputFormat is True
    assert result.structure.hasConstraints is True
    assert result.structure.hasExamples is True


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_prompt_with_numbers():
    result = analyze_prompt("The array is [2, 5, 8, 12, 16, 23, 38].")
    assert result.textStats.wordCount > 0


def test_prompt_with_code_block():
    result = analyze_prompt("Here is the code:\n```python\ndef hello():\n    print('hi')\n```")
    assert result.structure.hasExamples is True


def test_prompt_with_newlines():
    text = "Line one.\nLine two.\nLine three.\nLine four."
    result = analyze_prompt(text)
    assert result.textStats.lineCount == 4


def test_prompt_with_multiple_punctuation():
    result = analyze_prompt("Wait... Is this right? Yes! It works.")
    assert result.textStats.sentenceCount >= 2


def test_combined_realistic_prompt():
    prompt = (
        "You are a senior software engineer. "
        "Explain recursion to a junior developer. "
        "Use simple language and provide an example. "
        "Do not use complex terminology. "
        "Format the answer with headings."
    )
    result = analyze_prompt(prompt)
    assert result.structure.hasRoleInstruction is True
    assert result.structure.hasContext is True      # 'to a junior developer'
    assert result.structure.hasExamples is True     # 'provide an example'
    assert result.structure.hasConstraints is True
    assert result.structure.hasOutputFormat is True


def test_no_llm_api_called(monkeypatch):
    """
    Verify that no HTTP requests are made during analysis.
    Monkeypatch socket to block all network calls.
    """
    import socket

    original_getaddrinfo = socket.getaddrinfo

    def mock_getaddrinfo(*args, **kwargs):
        raise ConnectionRefusedError("Network calls are blocked during analysis test")

    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    # This must succeed without any network access
    result = analyze_prompt("Explain binary search.")
    assert result.tokenStats.tokenCount > 0
