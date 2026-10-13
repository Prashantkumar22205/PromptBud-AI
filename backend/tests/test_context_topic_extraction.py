"""
tests/test_context_topic_extraction.py

Unit tests for Module 3 Context Topic Extraction (SpacyContextExtractor).

Verifies candidate normalization, generic phrase filtering, deterministic scoring,
and evidence retention across standard and edge-case prompts.
"""

import pytest
from app.semantic.context.extractor import SpacyContextExtractor
from app.semantic.types import ContextResult, SemanticField


@pytest.fixture
def extractor():
    return SpacyContextExtractor()


def test_clear_technical_topic(extractor):
    """Case A: Clear technical topic in standard instruction."""
    prompt = "Explain binary search in C++ for a beginner."
    res: ContextResult = extractor.extract(prompt)

    assert res.topic is not None
    assert res.topic.value == "binary search"
    assert res.topic.confidence >= 0.40
    assert res.topic.method == "spacy_noun_phrase_scored"
    assert "binary search" in res.topic.evidence.lower()


def test_technical_topic_with_surrounding_context(extractor):
    """Case B: Technical topic embedded in long conversational/interview context."""
    prompt = "I am preparing for my DSA interview and I need you to explain how binary search works."
    res: ContextResult = extractor.extract(prompt)

    assert res.topic is not None
    assert res.topic.value == "binary search"
    assert res.topic.value not in ("i", "my dsa interview", "dsa interview")


def test_generic_introductory_phrase_filtered(extractor):
    """Case C: Generic introductory phrase ('our guide') should be filtered."""
    prompt = "Our guide wants us to explain the methodology for prompt optimization."
    res: ContextResult = extractor.extract(prompt)

    assert res.topic is not None
    assert res.topic.value != "our guide"
    assert "prompt optimization" in res.topic.value


def test_reference_heavy_prompt(extractor):
    """Case D: Reference-heavy prompt ('this', 'above') should not select generic references."""
    prompt = "Explain this algorithm and compare it with the above approach."
    res: ContextResult = extractor.extract(prompt)

    if res.topic is not None:
        assert res.topic.value not in ("this", "above", "the above approach", "it")
        assert res.topic.value == "algorithm"


def test_multiple_noun_phrases(extractor):
    """Case E: Multiple technical noun phrases — selects primary technical concept."""
    prompt = "Compare TF-IDF and Sentence Transformer embeddings for intent classification."
    res: ContextResult = extractor.extract(prompt)

    assert res.topic is not None
    assert res.topic.value in (
        "intent classification",
        "sentence transformer embeddings",
        "tf-idf and sentence transformer embeddings",
    )


def test_no_identifiable_topic(extractor):
    """Case F: Short, reference-only prompt with no identifiable technical topic."""
    prompt = "Do it again."
    res: ContextResult = extractor.extract(prompt)

    assert res.topic is None


def test_long_guide_situation_prompt(extractor):
    """User prompt problem example: 'our guide' must not be selected as topic."""
    prompt = "now the situation that our guide wants our final methodology of how are we going to go with the project for prompt optimization..."
    res: ContextResult = extractor.extract(prompt)

    if res.topic is not None:
        assert res.topic.value != "our guide"
        assert res.topic.value != "the situation"
        assert "prompt optimization" in res.topic.value


def test_existing_context_fields_unaffected(extractor):
    """Regression test: verify domain, audience, language, priorKnowledge work."""
    prompt = "Write a Python script for a beginner to connect to PostgreSQL database."
    res: ContextResult = extractor.extract(prompt)

    assert res.domain is not None
    assert res.domain.value in ("database", "software_engineering", "computer_science")

    assert res.audience is not None
    assert res.audience.value == "beginner"

    assert res.programmingLanguage is not None
    assert res.programmingLanguage.value.lower() == "python"
