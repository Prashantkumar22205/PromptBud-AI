"""
tests/test_requirements_extraction.py

Unit tests for Module 3 Requirement Extraction (SpacyRequirementExtractor).
Tests non-LLM dependency parsing, multi-verb clause traversal, pronoun rejection,
question-style requirement parsing, and long methodology prompt extraction.
"""

import pytest
from app.semantic.requirements.extractor import SpacyRequirementExtractor


@pytest.fixture
def extractor():
    return SpacyRequirementExtractor()


def test_1_explain_binary_search(extractor):
    """TEST 1: Explain binary search in C++ for a beginner."""
    prompt = "Explain binary search in C++ for a beginner."
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 1
    req = reqs[0]
    assert req.action == "explain"
    assert "binary search" in req.object.lower()
    assert req.mandatory is True
    assert req.object.lower() not in ("he", "it", "this", "we", "you")


def test_2_compare_tfidf_sentence_transformers(extractor):
    """TEST 2: Compare TF-IDF and Sentence Transformer embeddings for intent classification."""
    prompt = "Compare TF-IDF and Sentence Transformer embeddings for intent classification."
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 1
    req = reqs[0]
    assert req.action == "compare"
    assert "tf-idf" in req.object.lower() or "intent classification" in req.object.lower()
    assert req.object.lower() not in ("he", "it", "this", "we", "you")


def test_3_give_methodology_details(extractor):
    """TEST 3: Give all details of the methodology."""
    prompt = "Give all details of the methodology."
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 1
    req = reqs[0]
    assert req.action in ("provide", "give")
    assert "details of the methodology" in req.object.lower() or "detail" in req.object.lower()
    assert req.mandatory is True


def test_4_multi_verb_clause_requirements(extractor):
    """TEST 4: Explain the methodology and calculate the accuracy."""
    prompt = "Explain the methodology and calculate the accuracy."
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 2
    actions = [r.action.lower() for r in reqs]
    objects = [(r.object or "").lower() for r in reqs]
    assert "explain" in actions
    assert "calculate" in actions
    assert any("methodology" in obj for obj in objects)
    assert any("accuracy" in obj for obj in objects)


def test_5_question_how_to_calculate(extractor):
    """TEST 5: How do I calculate accuracy?"""
    prompt = "How do I calculate accuracy?"
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 1
    req = reqs[0]
    assert req.action == "calculate"
    assert req.object.lower() == "accuracy"


def test_6_question_what_formula(extractor):
    """TEST 6: What formula should I use?"""
    prompt = "What formula should I use?"
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 1
    req = reqs[0]
    assert req.action in ("provide", "identify", "use")
    assert "formula" in req.object.lower()


def test_7_ambiguous_do_it_again(extractor):
    """TEST 7: Do it again. (Pronoun 'it' must be rejected as requirement object)"""
    prompt = "Do it again."
    reqs = extractor.extract(prompt)
    for req in reqs:
        assert (req.object or "").lower() != "it"


def test_8_long_methodology_prompt(extractor):
    """TEST 8: Long methodology prompt from project."""
    prompt = (
        "now the situation that our guide wants our final methodology of how are we going to go with the project "
        "because previously we discussed and he had doubt on the project where we are using LLM for analysis and "
        "for everything, so he gives two options... now you have to give all detail of the methodology that we have "
        "decided, explain every detail so if he asks any question how are we calculating this, where the formula "
        "comes from and what research we can do with and the objective we can achieve."
    )
    reqs = extractor.extract(prompt)
    assert len(reqs) >= 2
    objects = [(r.object or "").lower() for r in reqs]

    # Must NOT select pronouns "he", "we", "you", "it", "this" as objects
    for r in reqs:
        assert r.object.lower() not in ("he", "we", "you", "it", "this")

    # Semantic objects should cover methodology details, calculations, formulas, or research
    joined_objects = " ".join(objects)
    assert any(term in joined_objects for term in ("methodology", "detail", "formula", "calculat", "research", "objective"))
