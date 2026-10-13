"""
tests/test_constraint_extraction.py

Unit tests for Module 3 Constraint Extraction (RuleConstraintExtractor).

Tests cover:
  - Format constraints   (table, bullet list, JSON, step by step)
  - Style constraints    (simple language, easy language, briefly)
  - Length constraints   (numeric limits, brevity adverbs)
  - Quantity constraints (5 examples, three methods)
  - Language constraints (Hindi, French, etc.)
  - Restriction constraints (do not use recursion, without LLM)
  - No false positives   (C++ not a language constraint, no invented constraints)
  - Real-world methodology prompt

Important constraint-class distinctions enforced:
  - REQUIREMENT = WHAT should be done
  - CONSTRAINT  = HOW / UNDER WHAT CONDITIONS it should be done
  - Programming language (C++) → context.programmingLanguage, NOT a constraint
  - Audience ("for a beginner") → context.audience, NOT a constraint
"""

import pytest
from app.semantic.constraints.extractor import RuleConstraintExtractor
from app.semantic.semantic_analyzer import analyze_semantics


@pytest.fixture
def extractor():
    return RuleConstraintExtractor()


# ---------------------------------------------------------------------------
# TEST 1: style constraint "simple language", C++ must NOT be a lang constraint
# ---------------------------------------------------------------------------

def test_1_simple_language_style(extractor):
    """
    Input: 'Explain binary search in C++ for a beginner in simple language.'
    Expected constraints: style → simple language
    NOT expected: C++ as a natural-language constraint
    """
    prompt = "Explain binary search in C++ for a beginner in simple language."
    constraints = extractor.extract(prompt)

    # Must detect style constraint
    style = [c for c in constraints if c.type == "style"]
    assert len(style) >= 1
    assert any("simple" in c.evidence.lower() for c in style)

    # C++ must NOT appear as a type=language constraint
    lang_constraints = [c for c in constraints if c.type == "language"]
    for lc in lang_constraints:
        assert "c++" not in lc.evidence.lower(), (
            "C++ is a programming language, not a natural-language output constraint"
        )

    # Confidence in range
    for c in constraints:
        assert 0.0 <= c.confidence <= 1.0
        assert c.evidence is not None


def test_1_end_to_end_cpp_not_in_constraints():
    """End-to-end: C++ must NOT appear as a language constraint in full analysis."""
    result = analyze_semantics(
        "Explain binary search in C++ for a beginner in simple language."
    )
    lang_c = [c for c in result.constraints if c.type == "language"]
    for lc in lang_c:
        assert "c++" not in lc.evidence.lower()
    # Audience stays in context
    assert result.context.audience is not None
    assert result.context.audience.value == "beginner"
    # C++ stays in context.programmingLanguage
    assert result.context.programmingLanguage is not None
    assert result.context.programmingLanguage.value == "C++"


# ---------------------------------------------------------------------------
# TEST 2: quantity constraint "5 examples"
# ---------------------------------------------------------------------------

def test_2_quantity_5_examples(extractor):
    """
    Input: 'Give me 5 examples of prompt optimization.'
    Expected: quantity constraint with value 5
    """
    prompt = "Give me 5 examples of prompt optimization."
    constraints = extractor.extract(prompt)

    quantity = [c for c in constraints if c.type == "quantity"]
    assert len(quantity) >= 1, "Should detect quantity constraint"
    assert "5" in quantity[0].description or "5" in quantity[0].evidence


# ---------------------------------------------------------------------------
# TEST 3: length constraint numeric limit
# ---------------------------------------------------------------------------

def test_3_length_less_than_500_words(extractor):
    """
    Input: 'Explain the methodology in less than 500 words.'
    Expected: length constraint with maxWords=500
    """
    prompt = "Explain the methodology in less than 500 words."
    constraints = extractor.extract(prompt)

    length = [c for c in constraints if c.type == "length"]
    assert len(length) >= 1, "Should detect length constraint"
    assert length[0].maxWords == 500


# ---------------------------------------------------------------------------
# TEST 4: format constraint "table"
# ---------------------------------------------------------------------------

def test_4_format_table(extractor):
    """
    Input: 'Compare E1 and E2 in a table.'
    Expected: format constraint with outputFormat='table'
    """
    prompt = "Compare E1 and E2 in a table."
    constraints = extractor.extract(prompt)

    format_c = [c for c in constraints if c.type == "format"]
    assert len(format_c) >= 1, "Should detect format constraint"
    assert format_c[0].outputFormat == "table"


# ---------------------------------------------------------------------------
# TEST 5: restriction "without using an LLM"
# ---------------------------------------------------------------------------

def test_5_restriction_without_llm(extractor):
    """
    Input: 'Explain this without using an LLM.'
    Expected: restriction constraint
    Note: 'this' must NOT become a meaningful requirement object.
    """
    prompt = "Explain this without using an LLM."
    constraints = extractor.extract(prompt)

    restriction = [c for c in constraints if c.type in ("restriction", "forbidden")]
    assert len(restriction) >= 1, "Should detect restriction constraint"
    assert "llm" in restriction[0].evidence.lower() or "using" in restriction[0].evidence.lower()


# ---------------------------------------------------------------------------
# TEST 6: format constraint "step by step"
# ---------------------------------------------------------------------------

def test_6_format_step_by_step(extractor):
    """
    Input: 'Give the answer step by step.'
    Expected: format constraint (step_by_step)
    """
    prompt = "Give the answer step by step."
    constraints = extractor.extract(prompt)

    fmt = [c for c in constraints if c.type == "format"]
    assert len(fmt) >= 1, "Should detect step-by-step format constraint"
    assert fmt[0].outputFormat == "step_by_step"


# ---------------------------------------------------------------------------
# TEST 7: natural language constraint "Hindi"
# ---------------------------------------------------------------------------

def test_7_language_hindi(extractor):
    """
    Input: 'Answer in Hindi.'
    Expected: language constraint with targetLanguage='Hindi'
    No duplicate — only one language constraint.
    """
    prompt = "Answer in Hindi."
    constraints = extractor.extract(prompt)

    lang = [c for c in constraints if c.type == "language"]
    assert len(lang) >= 1, "Should detect language constraint"
    # Deduplication: must not return two entries for the same evidence
    assert len(lang) == 1, f"Expected exactly 1 language constraint, got {len(lang)}: {[c.evidence for c in lang]}"
    assert lang[0].targetLanguage == "Hindi"


# ---------------------------------------------------------------------------
# TEST 8: length/style "briefly"
# ---------------------------------------------------------------------------

def test_8_briefly_constraint(extractor):
    """
    Input: 'Explain the algorithm briefly.'
    Expected: length or style constraint for 'briefly'
    """
    prompt = "Explain the algorithm briefly."
    constraints = extractor.extract(prompt)

    matched = [c for c in constraints if c.type in ("length", "style")]
    assert len(matched) >= 1, "Should detect 'briefly' as a length or style constraint"
    evidences = [c.evidence.lower() for c in matched]
    assert any("briefly" in ev for ev in evidences)


# ---------------------------------------------------------------------------
# TEST 9: restriction "do not use recursion"
# ---------------------------------------------------------------------------

def test_9_restriction_no_recursion(extractor):
    """
    Input: 'Do not use recursion.'
    Expected: restriction constraint, NO duplicates
    """
    prompt = "Do not use recursion."
    constraints = extractor.extract(prompt)

    restriction = [c for c in constraints if c.type in ("restriction", "forbidden")]
    assert len(restriction) >= 1, "Should detect restriction constraint"
    # Must not produce duplicates for same logical evidence
    assert len(restriction) == 1, (
        f"Expected exactly 1 restriction, got {len(restriction)}: "
        f"{[c.evidence for c in restriction]}"
    )
    assert "recursion" in restriction[0].evidence.lower()


# ---------------------------------------------------------------------------
# TEST 10: no false positive constraints
# ---------------------------------------------------------------------------

def test_10_no_false_positive_constraints(extractor):
    """
    Input: 'Explain binary search.'
    Expected: no constraints (none invented)
    """
    prompt = "Explain binary search."
    constraints = extractor.extract(prompt)
    assert constraints == [], (
        f"Should not invent constraints for a plain prompt, got: "
        f"{[(c.type, c.evidence) for c in constraints]}"
    )


# ---------------------------------------------------------------------------
# TEST 11: methodology prompt — only extract EXPLICIT constraints
# ---------------------------------------------------------------------------

def test_11_methodology_prompt_explicit_constraints_only(extractor):
    """
    The real project methodology prompt.
    Only extract explicit constraint-like instructions.
    'our guide wants...' does NOT become a constraint.
    """
    prompt = (
        "now the situation that our guide wants our final methodology of how are we going "
        "to go with the project because previously we discussed and he had doubt on the "
        "project where we are using LLM for analysis and for everything, so he gives two "
        "options... now you have to give all detail of the methodology that we have decided, "
        "explain every detail so if he asks any question how are we calculating this, where "
        "the formula comes from and what research we can do with and the objective we can achieve."
    )
    constraints = extractor.extract(prompt)

    # "our guide wants" must NOT produce a constraint
    for c in constraints:
        assert "our guide" not in c.evidence.lower(), (
            "'our guide wants' should not be extracted as a constraint"
        )

    # If any constraints are extracted they must have real evidence
    for c in constraints:
        assert c.evidence is not None
        assert len(c.evidence.strip()) > 0
        assert 0.0 <= c.confidence <= 1.0


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------

def test_evidence_always_in_prompt(extractor):
    """Every extracted constraint evidence must be a substring of the prompt."""
    prompts = [
        "Explain binary search in C++ for a beginner in simple language.",
        "Give me 5 examples of prompt optimization.",
        "Explain the methodology in less than 500 words.",
        "Compare E1 and E2 in a table.",
        "Explain this without using an LLM.",
        "Give the answer step by step.",
        "Answer in Hindi.",
        "Explain the algorithm briefly.",
        "Do not use recursion.",
    ]
    for prompt in prompts:
        for c in extractor.extract(prompt):
            assert c.evidence.strip() in prompt or c.evidence.strip().rstrip(".") in prompt, (
                f"Evidence {c.evidence!r} not found in prompt {prompt!r}"
            )


def test_confidence_range(extractor):
    """Confidence must always be in [0.0, 1.0]."""
    prompt = (
        "Explain binary search in simple language, in under 200 words, "
        "as a table, answer in Hindi, without using an LLM. "
        "Give me 5 examples."
    )
    for c in extractor.extract(prompt):
        assert 0.0 <= c.confidence <= 1.0, f"Confidence out of range: {c.confidence}"


def test_no_duplicate_constraints_for_same_evidence(extractor):
    """Same evidence phrase must not generate two constraint objects."""
    prompt = "Explain in simple language."
    constraints = extractor.extract(prompt)
    evidences_norm = [
        c.evidence.lower().strip().rstrip(".,;!?") for c in constraints
    ]
    assert len(evidences_norm) == len(set(evidences_norm)), (
        f"Duplicate constraint evidences: {evidences_norm}"
    )


def test_format_json(extractor):
    """'Return JSON' should produce format=json."""
    prompt = "Analyze the data and return JSON."
    constraints = extractor.extract(prompt)
    fmt = [c for c in constraints if c.type == "format"]
    assert len(fmt) >= 1
    assert fmt[0].outputFormat == "json"


def test_restriction_dont_use_libraries(extractor):
    """'Do not use external libraries' → restriction."""
    prompt = "Implement quicksort. Do not use external libraries."
    constraints = extractor.extract(prompt)
    restriction = [c for c in constraints if c.type in ("restriction", "forbidden")]
    assert len(restriction) >= 1
    assert "librar" in restriction[0].evidence.lower() or "external" in restriction[0].evidence.lower()


def test_style_easy_language(extractor):
    """'in easy language' → style constraint."""
    prompt = "Explain neural networks in easy language."
    constraints = extractor.extract(prompt)
    style = [c for c in constraints if c.type == "style"]
    assert len(style) >= 1
    assert any("easy" in c.evidence.lower() for c in style)


def test_quantity_word_number(extractor):
    """'provide three methods' → quantity constraint with value 3."""
    prompt = "Provide three methods to solve this problem."
    constraints = extractor.extract(prompt)
    qty = [c for c in constraints if c.type == "quantity"]
    assert len(qty) >= 1
    assert "3" in qty[0].description or "three" in qty[0].evidence.lower()
