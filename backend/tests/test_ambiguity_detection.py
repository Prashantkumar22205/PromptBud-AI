"""Regression coverage for the deterministic ambiguity detector."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.semantic.ambiguity.detector import RuleAmbiguityDetector
from app.semantic.semantic_analyzer import analyze_semantics


detector = RuleAmbiguityDetector()
client = TestClient(app)


def types(prompt):
    return {issue.type for issue in detector.detect(prompt)}


@pytest.mark.parametrize("prompt", ["Explain this.", "Fix it.", "Do it again.", "Compare it with that."])
def test_unresolved_references_are_detected(prompt):
    assert "referential_ambiguity" in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Explain binary search and compare it with linear search.",
    "Explain binary search.",
    "Explain binary search in C++ for a beginner.",
])
def test_locally_grounded_references_and_specific_tasks_are_not_ambiguous(prompt):
    assert "referential_ambiguity" not in types(prompt)


@pytest.mark.parametrize("prompt", ["Make it better.", "Give me a good answer."])
def test_vague_instructions_are_detected(prompt):
    assert "vague_instruction" in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Optimize this prompt to reduce unnecessary words.",
    "Optimize the SQL query to reduce execution time.",
    "Find the best algorithm for sorting 1 million integers.",
    "Explain in detail using simple language.",
    "Give 5 examples in a table.",
])
def test_specific_objectives_and_compatible_constraints_are_not_vague(prompt):
    assert "vague_instruction" not in types(prompt)


@pytest.mark.parametrize("prompt", ["Give me code.", "Translate this.", "Analyze the data.", "Create a presentation."])
def test_missing_essential_information_is_detected(prompt):
    assert "missing_essential_information" in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Write a Python function to reverse a string.",
    "Translate 'Hello' into French.",
    "Create a 5-slide presentation about cloud computing.",
])
def test_sufficiently_specified_tasks_are_not_missing_essential_information(prompt):
    assert "missing_essential_information" not in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Explain this in 20 words with exhaustive detail.",
    "Give one sentence and ten detailed paragraphs.",
])
def test_conflicting_constraints_are_detected(prompt):
    assert "conflicting_constraints" in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Explain in detail using simple language.",
    "Explain briefly in 100 words.",
    "Give 5 examples in a table.",
])
def test_compatible_constraints_are_not_conflicting(prompt):
    assert "conflicting_constraints" not in types(prompt)


@pytest.mark.parametrize("prompt", ["Explain the whole thing.", "Analyze everything.", "Fix everything."])
def test_unbounded_scope_is_detected(prompt):
    assert "unclear_scope" in types(prompt)


@pytest.mark.parametrize("prompt", [
    "Explain the complete methodology for intent classification.",
    "Analyze all columns in the dataset.",
    "Explain the methodology and calculate its accuracy.",
])
def test_named_scope_is_not_unclear(prompt):
    assert "unclear_scope" not in types(prompt)


@pytest.mark.parametrize("prompt", ["", "   ", ".", "? ", "x", "print('hello')", "https://example.com"])
def test_edge_inputs_never_crash(prompt):
    assert isinstance(detector.detect(prompt), list)


def test_mixed_ambiguity_returns_multiple_categories():
    assert {"referential_ambiguity", "vague_instruction"} <= types("Optimize it and make it better.")
    assert {"missing_essential_information", "unclear_scope"} <= types("Give me code and explain everything.")


@pytest.mark.parametrize("prompt,expected", [
    ("Explain binary search in C++ for a beginner.", False),
    ("Explain this.", True),
    ("Give me code.", True),
    ("Make it better.", True),
    ("Explain this in 20 words with exhaustive detail.", True),
    ("Explain binary search in detail using simple language.", False),
    ("Explain the complete methodology for intent classification.", False),
    ("Do it again.", True),
])
def test_semantic_api_manual_cases(prompt, expected):
    response = client.post("/api/semantic-analyze", json={"prompt": prompt})
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert bool(analysis["ambiguities"]) is expected


def test_semantic_profile_includes_ambiguities_without_schema_change():
    result = analyze_semantics("Explain this.")
    assert result.ambiguities
    assert "ambiguities" in result.model_dump()
