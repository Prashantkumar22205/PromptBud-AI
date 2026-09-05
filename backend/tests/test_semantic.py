"""test_semantic.py

Comprehensive test suite for Module 3 (Semantic Analysis).

Tests cover:
  Section A: Intent Classification
  Section B: Context Extraction
  Section C: Requirements Extraction
  Section D: Constraints Extraction
  Section E: Ambiguity Detection
  Section F: API Endpoint (/api/semantic-analyze)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.semantic.semantic_analyzer import analyze_semantics, SemanticAnalyzer
from app.semantic.types import (
    SemanticAnalysis,
    IntentResult,
    ContextResult,
    Requirement,
    Constraint,
    Ambiguity,
)

client = TestClient(app)


# ============================================================================
# Section A: Intent Classification Tests
# ============================================================================

def test_intent_explanation():
    """Test intent classification for explanation prompt."""
    result = analyze_semantics("Explain how bubble sort works step by step.")
    assert result.intent.primaryIntent in ("educational_explanation", "code_explanation", "explanation")
    assert result.intent.confidence >= 0.10
    assert result.intent.method == "tfidf_logistic_regression"


def test_intent_code_generation():
    """Test intent classification for code generation prompt."""
    result = analyze_semantics("Write a Python script to parse JSON files.")
    assert result.intent.primaryIntent == "code_generation"
    assert result.intent.confidence >= 0.10


def test_intent_debugging():
    """Test intent classification for debugging prompt."""
    result = analyze_semantics("Why is my Python function throwing a TypeError?")
    assert result.intent.primaryIntent in ("code_debugging", "debugging")
    assert result.intent.confidence >= 0.10


def test_intent_summarization():
    """Test intent classification for summarization prompt."""
    result = analyze_semantics("Summarize the key points of the attached article.")
    assert result.intent.primaryIntent == "summarization"
    assert result.intent.confidence >= 0.10



def test_intent_fallback_other():
    """Test fallback to 'other' for obscure or low confidence prompts."""
    result = analyze_semantics("xyz123 qwerty foo bar baz")
    # Low confidence should fallback to 'other'
    assert result.intent.primaryIntent in ("other", "creative_writing")
    assert isinstance(result.intent.confidence, float)


def test_intent_schema():
    """Test intent result structure."""
    result = analyze_semantics("Refactor this C++ function to optimize performance.")
    assert isinstance(result.intent, IntentResult)
    assert hasattr(result.intent, "primaryIntent")
    assert hasattr(result.intent, "confidence")
    assert hasattr(result.intent, "method")


# ============================================================================
# Section B: Context Extraction Tests
# ============================================================================

def test_context_topic_extraction():
    """Test topic extraction from prompt."""
    result = analyze_semantics("Explain quantum computing algorithms in simple terms.")
    assert result.context.topic is not None
    assert len(result.context.topic.value) > 0


def test_context_domain_detection():
    """Test domain classification."""
    result = analyze_semantics("Write a SQL query to join two tables and filter by date.")
    assert result.context.domain is not None
    assert result.context.domain.value in ("database", "software_engineering", "computer_science")


def test_context_target_audience():
    """Test target audience detection."""
    result = analyze_semantics("Explain bubble sort to a beginner.")
    assert result.context.audience is not None
    assert result.context.audience.value == "beginner"


def test_context_programming_language():
    """Test programming language detection."""
    result = analyze_semantics("Write a Python function to read CSV files.")
    assert result.context.programmingLanguage is not None
    assert result.context.programmingLanguage.value.lower() == "python"


def test_context_prior_knowledge():
    """Test prior knowledge detection."""
    result = analyze_semantics("Explain neural networks assuming no math background.")
    assert isinstance(result.context.priorKnowledge, list)


# ============================================================================
# Section C: Requirements Extraction Tests
# ============================================================================

def test_requirements_action_verbs():
    """Test requirement extraction from action verbs."""
    result = analyze_semantics("Explain bubble sort and provide Python code.")
    assert len(result.requirements) >= 1
    req_actions = [r.action.lower() for r in result.requirements]
    assert "explain" in req_actions or "provide" in req_actions


def test_requirements_desired_output():
    """Test requirement extraction for desired outputs."""
    result = analyze_semantics("Provide code and a step by step walkthrough.")
    assert len(result.requirements) >= 1
    for req in result.requirements:
        assert req.confidence > 0.0
        assert req.evidence is not None


def test_requirements_complexity_detail():
    """Test requirement extraction for specific details requested."""
    result = analyze_semantics("Analyze time complexity and space complexity of merge sort.")
    assert len(result.requirements) >= 1
    req_evidences = [(r.evidence or "").lower() for r in result.requirements]
    assert any("complexity" in ev or "analyze" in ev for ev in req_evidences)


def test_requirements_empty_safe():
    """Test requirements extraction on short text."""
    result = analyze_semantics("Hello")
    assert isinstance(result.requirements, list)


# ============================================================================
# Section D: Constraints Extraction Tests
# ============================================================================

def test_constraints_length():
    """Test length constraint extraction."""
    result = analyze_semantics("Explain bubble sort in under 100 words.")
    length_constraints = [c for c in result.constraints if c.type == "length"]
    assert len(length_constraints) >= 1
    assert length_constraints[0].maxWords == 100


def test_constraints_format():
    """Test format constraint extraction."""
    result = analyze_semantics("List top 5 sorting algorithms in a bulleted list.")
    format_constraints = [c for c in result.constraints if c.type == "format"]
    assert len(format_constraints) >= 1
    assert format_constraints[0].outputFormat == "bullet_list"


def test_constraints_tone_style():
    """Test style/tone constraint extraction."""
    result = analyze_semantics("Explain machine learning in simple terms for beginners.")
    style_constraints = [c for c in result.constraints if c.type in ("style", "tone")]
    assert len(style_constraints) >= 1


def test_constraints_negative():
    """Test negative constraint extraction."""
    result = analyze_semantics("Implement quicksort. Do not use external libraries.")
    negative_constraints = [c for c in result.constraints if c.type in ("negative", "forbidden")]
    assert len(negative_constraints) >= 1
    assert "libraries" in negative_constraints[0].description.lower() or "external" in negative_constraints[0].description.lower()



def test_constraints_language():
    """Test target language constraint extraction."""
    result = analyze_semantics("Translate this sentence into French.")
    lang_constraints = [c for c in result.constraints if c.type == "language"]
    assert len(lang_constraints) >= 1
    assert lang_constraints[0].targetLanguage == "French"


# ============================================================================
# Section E: Ambiguity Detection Tests
# ============================================================================

def test_ambiguity_conflicting_constraints():
    """Test detection of conflicting constraints."""
    result = analyze_semantics("Write a very short summary in exhaustive detail.")
    conflicts = [a for a in result.ambiguities if a.type == "conflicting_constraints"]
    assert len(conflicts) >= 1
    assert conflicts[0].severity == "high"


def test_ambiguity_underspecified_object():
    """Test detection of underspecified phrases."""
    result = analyze_semantics("Make my python code better and fix it.")
    underspecified = [a for a in result.ambiguities if a.type == "underspecified_object"]
    assert len(underspecified) >= 1


def test_ambiguity_normal_prompt():
    """Test normal prompt has no high-severity conflicts."""
    result = analyze_semantics("Explain how binary search works in Python with O(log n) time complexity.")
    high_severity_ambiguities = [a for a in result.ambiguities if a.severity == "high"]
    assert len(high_severity_ambiguities) == 0


# ============================================================================
# Section F: API Endpoint Tests
# ============================================================================

def test_api_semantic_analyze_success():
    """Test POST /api/semantic-analyze returns 200 and structured response."""
    response = client.post(
        "/api/semantic-analyze",
        json={"prompt": "Explain bubble sort to a beginner in Python under 200 words."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "analysis" in data
    analysis = data["analysis"]
    assert analysis["intent"]["primaryIntent"] in ("educational_explanation", "code_explanation", "explanation")
    assert analysis["context"]["audience"]["value"] == "beginner"
    assert analysis["context"]["programmingLanguage"]["value"].lower() == "python"
    assert len(analysis["constraints"]) >= 1



def test_api_semantic_analyze_empty_prompt():
    """Test POST /api/semantic-analyze returns 400 for empty prompt."""
    response = client.post(
        "/api/semantic-analyze",
        json={"prompt": "   "},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["code"] == "EMPTY_PROMPT"


def test_api_semantic_analyze_missing_body():
    """Test POST /api/semantic-analyze returns 422 for missing body."""
    response = client.post(
        "/api/semantic-analyze",
        json={},
    )
    assert response.status_code == 422


def test_api_health_still_works():
    """Verify /health endpoint remains functional."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
