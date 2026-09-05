"""
tests/test_tokenizer.py

Unit tests for the tokenizer abstraction layer.
"""

from __future__ import annotations

import pytest

from app.analysis.tokenizer import (
    BaseTokenizer,
    TiktokenTokenizer,
    get_tokenizer,
    TOKENIZER_REGISTRY,
)


def test_tiktoken_tokenizer_returns_positive_count():
    t = TiktokenTokenizer()
    assert t.count_tokens("Hello, world!") > 0


def test_tiktoken_tokenizer_name():
    t = TiktokenTokenizer()
    assert "tiktoken" in t.name
    assert "cl100k_base" in t.name


def test_tiktoken_empty_string():
    t = TiktokenTokenizer()
    assert t.count_tokens("") == 0


def test_tiktoken_longer_text_has_more_tokens():
    t = TiktokenTokenizer()
    short = t.count_tokens("Hi.")
    long = t.count_tokens("Hi. " * 100)
    assert long > short


def test_get_tokenizer_returns_base_tokenizer():
    t = get_tokenizer()
    assert isinstance(t, BaseTokenizer)


def test_get_tokenizer_default_is_tiktoken():
    t = get_tokenizer()
    assert isinstance(t, TiktokenTokenizer)


def test_get_tokenizer_caching():
    t1 = get_tokenizer()
    t2 = get_tokenizer()
    assert t1 is t2  # same cached instance


def test_tokenizer_registry_contains_tiktoken():
    assert "tiktoken" in TOKENIZER_REGISTRY


def test_get_tokenizer_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown tokenizer key"):
        get_tokenizer("nonexistent_tokenizer_key")


def test_tokenizer_is_deterministic():
    t = TiktokenTokenizer()
    text = "The quick brown fox jumps over the lazy dog."
    count1 = t.count_tokens(text)
    count2 = t.count_tokens(text)
    assert count1 == count2


def test_whitespace_only_tokens():
    t = TiktokenTokenizer()
    # Whitespace still tokenises to something
    count = t.count_tokens("   \n   ")
    assert isinstance(count, int)
    assert count >= 0
