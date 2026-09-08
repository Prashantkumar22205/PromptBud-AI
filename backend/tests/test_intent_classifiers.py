"""
tests/test_intent_classifiers.py

Unit and integration test suite for Module 3 Intent Classifiers:
  - TfidfLogisticIntentClassifier (E1 Baseline)
  - SemanticIntentClassifier (E2 SentenceTransformer Embeddings)
  - Factory and Configuration Selection
"""

import math
import pytest
from app.core.config import settings
from app.semantic.intent.classifier import (
    BaseIntentClassifier,
    TfidfLogisticIntentClassifier,
    SemanticIntentClassifier,
    HybridIntentClassifier,
    get_intent_classifier,
)
from app.semantic.types import IntentResult


# Sample training data for unit tests
TRAIN_TEXTS = [
    "Explain binary search algorithm step by step.",
    "Teach me how quicksort works for beginners.",
    "Write a Python script to parse JSON files.",
    "Create a FastAPI backend endpoint in Python.",
    "Why is my Python function throwing KeyError?",
    "Debug this SQL query returning duplicates.",
    "Summarize this article in 3 bullet points.",
    "Give me a short summary of this report.",
]
TRAIN_LABELS = [
    "educational_explanation",
    "educational_explanation",
    "code_generation",
    "code_generation",
    "code_debugging",
    "code_debugging",
    "summarization",
    "summarization",
]


def test_e1_initialization_and_method():
    clf = TfidfLogisticIntentClassifier()
    assert clf.method == "tfidf_logistic_regression"
    assert isinstance(clf, BaseIntentClassifier)


def test_e2_initialization_and_method():
    clf = SemanticIntentClassifier()
    assert clf.method == "sentence_transformer_logistic_regression"
    assert isinstance(clf, BaseIntentClassifier)


def test_e3_initialization_and_method():
    clf = HybridIntentClassifier(alpha=0.5)
    assert clf.method == "hybrid_tfidf_semantic"
    assert isinstance(clf, BaseIntentClassifier)


def test_e3_alpha_validation():
    with pytest.raises(ValueError):
        HybridIntentClassifier(alpha=-0.1)
    with pytest.raises(ValueError):
        HybridIntentClassifier(alpha=1.1)


def test_e1_fit_and_predict():
    clf = TfidfLogisticIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)

    label, conf, all_scores = clf.predict("Write a Python function to sort a list.")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0
    assert isinstance(all_scores, dict)
    assert len(all_scores) == 4
    assert math.isclose(sum(all_scores.values()), 1.0, abs_tol=1e-2)
    assert math.isclose(conf, max(all_scores.values()), abs_tol=1e-4)


def test_e2_fit_and_predict():
    clf = SemanticIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)

    label, conf, all_scores = clf.predict("Write a Python script to process text.")
    assert isinstance(label, str)
    assert 0.0 <= conf <= 1.0
    assert isinstance(all_scores, dict)
    assert len(all_scores) == 4
    assert math.isclose(sum(all_scores.values()), 1.0, abs_tol=1e-2)
    assert math.isclose(conf, max(all_scores.values()), abs_tol=1e-4)


def test_e3_alpha_0_produces_e2_probabilities():
    e1 = TfidfLogisticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e2 = SemanticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e3 = HybridIntentClassifier(alpha=0.0, confidence_threshold=0.0, e1_clf=e1, e2_clf=e2)

    prompt = "Summarize this long report."
    _, _, scores_e2 = e2.predict(prompt)
    _, _, scores_e3 = e3.predict(prompt)

    for k in scores_e2:
        assert math.isclose(scores_e3[k], scores_e2[k], abs_tol=1e-3)


def test_e3_alpha_1_produces_e1_probabilities():
    e1 = TfidfLogisticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e2 = SemanticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e3 = HybridIntentClassifier(alpha=1.0, confidence_threshold=0.0, e1_clf=e1, e2_clf=e2)

    prompt = "Summarize this long report."
    _, _, scores_e1 = e1.predict(prompt)
    _, _, scores_e3 = e3.predict(prompt)

    for k in scores_e1:
        assert math.isclose(scores_e3[k], scores_e1[k], abs_tol=1e-3)


def test_e3_alpha_0_5_averages_probabilities():
    e1 = TfidfLogisticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e2 = SemanticIntentClassifier(confidence_threshold=0.0).fit(TRAIN_TEXTS, TRAIN_LABELS)
    e3 = HybridIntentClassifier(alpha=0.5, confidence_threshold=0.0, e1_clf=e1, e2_clf=e2)

    prompt = "Explain binary search in Python."
    _, _, scores_e1 = e1.predict(prompt)
    _, _, scores_e2 = e2.predict(prompt)
    lbl3, conf3, scores_e3 = e3.predict(prompt)

    for k in scores_e1:
        expected = (scores_e1[k] + scores_e2[k]) / 2.0
        assert math.isclose(scores_e3[k], expected, abs_tol=1e-3)

    assert math.isclose(sum(scores_e3.values()), 1.0, abs_tol=1e-2)
    assert math.isclose(conf3, max(scores_e3.values()), abs_tol=1e-4)
    assert lbl3 == max(scores_e3, key=scores_e3.get)


def test_classify_returns_intent_result_with_all_scores():
    clf = TfidfLogisticIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)

    res = clf.classify("Explain recursion to a student.")
    assert isinstance(res, IntentResult)
    assert res.primaryIntent in ("educational_explanation", "summarization", "other")
    assert res.method == "tfidf_logistic_regression"
    assert res.allScores is not None
    assert len(res.allScores) == 4
    assert math.isclose(sum(res.allScores.values()), 1.0, abs_tol=1e-3)


def test_e2_classify_returns_intent_result_with_all_scores():
    clf = SemanticIntentClassifier()
    clf.fit(TRAIN_TEXTS, TRAIN_LABELS)

    res = clf.classify("Explain recursion to a student.")
    assert isinstance(res, IntentResult)
    assert res.method == "sentence_transformer_logistic_regression"
    assert res.allScores is not None
    assert len(res.allScores) == 4
    assert math.isclose(sum(res.allScores.values()), 1.0, abs_tol=1e-3)


def test_deterministic_random_state():
    clf1 = TfidfLogisticIntentClassifier()
    clf1.fit(TRAIN_TEXTS, TRAIN_LABELS)
    l1, c1, _ = clf1.predict("Summarize this long report.")

    clf2 = TfidfLogisticIntentClassifier()
    clf2.fit(TRAIN_TEXTS, TRAIN_LABELS)
    l2, c2, _ = clf2.predict("Summarize this long report.")

    assert l1 == l2
    assert c1 == c2


def test_classifier_factory_tfidf():
    clf = get_intent_classifier("tfidf")
    assert clf.method == "tfidf_logistic_regression"


def test_classifier_factory_semantic():
    clf = get_intent_classifier("semantic")
    assert clf.method == "sentence_transformer_logistic_regression"


def test_classifier_factory_hybrid():
    clf = get_intent_classifier("hybrid")
    assert clf.method == "hybrid_tfidf_semantic"


def test_factory_respects_settings_default():
    clf = get_intent_classifier()
    expected_method = (
        "sentence_transformer_logistic_regression"
        if settings.intent_classifier_type == "semantic"
        else "hybrid_tfidf_semantic"
        if settings.intent_classifier_type == "hybrid"
        else "tfidf_logistic_regression"
    )
    assert clf.method == expected_method

