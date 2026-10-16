"""Tests for Module 5 metrics and its independent API contract."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.analysis.tokenizer import BaseTokenizer
from app.evaluation.evaluator import PromptEvaluator
from app.evaluation.metrics.constraint_metrics import calculate_constraint_preservation
from app.evaluation.metrics.requirement_metrics import calculate_requirement_preservation
from app.evaluation.metrics.semantic_metrics import SentenceTransformerSimilarity
from app.evaluation.metrics.token_metrics import calculate_token_metrics
from app.evaluation.metrics.constraint_metrics import calculate_constraint_preservation
from app.evaluation.types import EvaluationRequest
from app.main import app
from app.semantic.constraints.extractor import RuleConstraintExtractor
from app.semantic.context.extractor import SpacyContextExtractor
from app.semantic.requirements.extractor import SpacyRequirementExtractor
from app.semantic.types import Constraint, Requirement


class WordTokenizer(BaseTokenizer):
    @property
    def name(self) -> str:
        return "test/word"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeRequirementExtractor:
    def __init__(self, results_by_prompt):
        self.results_by_prompt = results_by_prompt

    def extract(self, prompt):
        return list(self.results_by_prompt.get(prompt, []))


class FakeConstraintExtractor:
    def __init__(self, results_by_prompt):
        self.results_by_prompt = results_by_prompt

    def extract(self, prompt):
        return list(self.results_by_prompt.get(prompt, []))


def requirement(action="explain", object_="binary search", mandatory=True):
    return Requirement(action=action, object=object_, mandatory=mandatory, confidence=0.9, method="test")


def constraint(type_="format", description="Output format: bullet list", **kwargs):
    return Constraint(type=type_, description=description, confidence=0.9, method="test", **kwargs)


def test_token_count_returns_counts_for_both_prompts():
    result = calculate_token_metrics("one two three", "one two", WordTokenizer())
    assert result.original_tokens == 3
    assert result.optimized_tokens == 2


def test_token_reduction_positive_case():
    result = calculate_token_metrics("one two three four", "one two", WordTokenizer())
    assert result.token_delta == 2
    assert result.token_reduction_percent == 50.0


def test_token_reduction_same_length_is_zero():
    result = calculate_token_metrics("one two", "three four", WordTokenizer())
    assert result.token_delta == 0
    assert result.token_reduction_percent == 0.0


def test_token_reduction_when_optimized_prompt_is_longer():
    result = calculate_token_metrics("short", "this optimized prompt is longer", WordTokenizer())
    assert result.token_delta == -4
    assert result.token_reduction_percent == -400.0


def test_zero_token_edge_case_is_finite_and_safe():
    result = calculate_token_metrics("", "", WordTokenizer())
    assert result.original_tokens == result.optimized_tokens == result.token_delta == 0
    assert result.token_reduction_percent == 0.0


class FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        vectors = {
            "identical": [1.0, 0.0],
            "different": [0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts])


def test_semantic_similarity_identical_prompts_is_one():
    metric = SentenceTransformerSimilarity("test-model", FakeEmbeddingModel())
    assert metric.calculate("identical", "identical").score == pytest.approx(1.0)


def test_semantic_similarity_different_prompts_is_lower_than_identical():
    metric = SentenceTransformerSimilarity("test-model", FakeEmbeddingModel())
    same = metric.calculate("identical", "identical").score
    different = metric.calculate("identical", "different").score
    assert 0.0 <= different <= 1.0
    assert different < same


def test_semantic_similarity_is_affine_normalized_cosine_and_encodes_both_together():
    class RecordingModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append((list(texts), kwargs))
            return np.asarray([[1.0, 0.0], [0.0, 1.0]])

    model = RecordingModel()
    score = SentenceTransformerSimilarity("one-model", model).calculate("original", "optimized").score
    assert score == pytest.approx(0.5)  # cosine(orthogonal vectors)=0, normalized from [-1,1] to [0,1]
    assert len(model.calls) == 1
    assert model.calls[0][0] == ["original", "optimized"]


def test_semantic_similarity_handles_empty_input_vectors():
    class ZeroVectorModel:
        def encode(self, texts, **kwargs):
            return np.zeros((2, 3))

    score = SentenceTransformerSimilarity("test-model", ZeroVectorModel()).calculate("", "").score
    assert score == 1.0


def test_requirement_preservation_when_all_requirements_remain():
    original = [requirement(), requirement("provide", "an example")]
    extractor = FakeRequirementExtractor({"original": original, "optimized": list(original)})
    result = calculate_requirement_preservation("original", "optimized", extractor)
    assert result.preserved_count == result.original_count == 2
    assert result.preservation_score == 1.0


def test_requirement_preservation_when_one_requirement_is_removed():
    extractor = FakeRequirementExtractor(
        {"original": [requirement(), requirement("provide", "an example")], "optimized": [requirement()]}
    )
    result = calculate_requirement_preservation("original", "optimized", extractor)
    assert result.preserved_count == 1
    assert result.original_count == 2
    assert result.preservation_score == 0.5


def test_requirement_matching_preserves_examples_when_folded_into_explain_object():
    extractor = FakeRequirementExtractor(
        {
            "Explain binary search and give 3 examples.": [
                requirement("explain", "binary search"),
                requirement("provide", "3 examples"),
            ],
            "Explain binary search with 3 examples.": [
                requirement("explain", "binary search with 3 examples")
            ],
        }
    )
    result = calculate_requirement_preservation(
        "Explain binary search and give 3 examples.",
        "Explain binary search with 3 examples.",
        extractor,
    )
    assert result.preserved_count == result.original_count == 2
    assert result.preservation_score == 1.0


def test_requirement_matching_detects_removed_example_requirement():
    extractor = FakeRequirementExtractor(
        {
            "Explain binary search and give 3 examples.": [
                requirement("explain", "binary search"),
                requirement("provide", "3 examples"),
            ],
            "Explain binary search.": [requirement("explain", "binary search")],
        }
    )
    result = calculate_requirement_preservation(
        "Explain binary search and give 3 examples.", "Explain binary search.", extractor
    )
    assert result.preserved_count == 1
    assert result.original_count == 2
    assert result.preservation_score == 0.5


def test_requirement_matching_is_one_to_one_for_multiple_requirements():
    requirements = [
        requirement("explain", "binary search"),
        requirement("compare", "linear search"),
        requirement("provide", "C++ code"),
    ]
    extractor = FakeRequirementExtractor({"original": requirements, "optimized": requirements})
    result = calculate_requirement_preservation("original", "optimized", extractor)
    assert result.preserved_count == result.original_count == 3


def test_requirement_paraphrase_case_exposes_module3_extractor_limit():
    # Module 3 emits an extra "walkthrough" requirement for the original and
    # object phrases that differ, so Module 5 cannot claim full preservation.
    extractor = FakeRequirementExtractor(
        {
            "Walk me through binary search.": [
                requirement("explain", "through binary search"),
                requirement("explain", "walkthrough"),
            ],
            "Explain how binary search works.": [requirement("explain", "how binary search works")],
        }
    )
    result = calculate_requirement_preservation(
        "Walk me through binary search.", "Explain how binary search works.", extractor
    )
    assert result.preserved_count == 1
    assert result.original_count == 2
    assert result.preservation_score == 0.5


@pytest.mark.parametrize(
    "original,optimized,expected_preserved,expected_total",
    [
        (
            "Explain binary search and give 3 examples.",
            "Explain binary search with 3 examples.",
            2,
            2,
        ),
        (
            "Explain binary search and give 3 examples.",
            "Explain binary search.",
            1,
            2,
        ),
        (
            "Explain binary search, compare it with linear search, and provide C++ code.",
            "Explain binary search, compare it with linear search, and provide C++ code.",
            3,
            3,
        ),
        (
            "Walk me through binary search.",
            "Explain how binary search works.",
            1,
            2,
        ),
    ],
)
def test_adversarial_requirement_cases_with_actual_module3_extractor(
    original, optimized, expected_preserved, expected_total
):
    result = calculate_requirement_preservation(original, optimized, SpacyRequirementExtractor())
    assert result.preserved_count == expected_preserved
    assert result.original_count == expected_total


def test_constraint_preservation_when_all_constraints_remain():
    original = [constraint(), constraint("length", "Length limit: maximum 100 words", maxWords=100)]
    extractor = FakeConstraintExtractor({"original": original, "optimized": list(original)})
    result = calculate_constraint_preservation("original", "optimized", extractor)
    assert result.preserved_count == result.original_count == 2
    assert result.preservation_score == 1.0


def test_constraint_preservation_when_structured_constraint_removed():
    original = [
        constraint("length", "Length limit: maximum 100 words", maxWords=100),
        constraint("format", "Output format: bullet list", outputFormat="bullet_list"),
    ]
    extractor = FakeConstraintExtractor({"original": original, "optimized": [original[0]]})
    result = calculate_constraint_preservation("original", "optimized", extractor)
    assert result.preserved_count == 1
    assert result.original_count == 2
    assert result.preservation_score == 0.5


@pytest.mark.parametrize(
    "original,optimized,expected",
    [
        ("Give 5 examples.", "Give 3 examples.", 0.0),
        ("Answer in a table.", "Answer in bullet points.", 0.0),
        ("Do not use recursion.", "Do not use recursion.", 1.0),
        ("Do not use recursion.", "Use recursion.", 0.0),
    ],
)
def test_constraint_audit_cases_use_module3_constraint_extractor(original, optimized, expected):
    result = calculate_constraint_preservation(original, optimized, RuleConstraintExtractor())
    assert result.preservation_score == expected


def test_programming_language_is_context_not_constraint():
    original = "Explain binary search in C++ in simple language."
    optimized = "Explain binary search in Python in simple language."
    context_extractor = SpacyContextExtractor()
    original_language = context_extractor.extract(original).programmingLanguage.value
    optimized_language = context_extractor.extract(optimized).programmingLanguage.value
    assert original_language == "C++"
    assert optimized_language == "Python"

    # The Module 3 constraint extractor deliberately excludes programming
    # languages, so this score covers only its extracted constraints.
    result = calculate_constraint_preservation(original, optimized, RuleConstraintExtractor())
    assert result.original_constraints
    assert result.preservation_score == 1.0


def test_using_simple_language_phrase_is_a_module3_extraction_gap():
    prompt = "Explain binary search in C++ using simple language."
    result = calculate_constraint_preservation(prompt, prompt, RuleConstraintExtractor())
    # Module 3's regex extractor does not recognize this exact phrasing, so a
    # vacuous 1.0 score does not establish that the style constraint was kept.
    assert result.original_count == 0
    assert result.preservation_score == 1.0


def test_ordinary_prompt_content_does_not_create_module3_constraints():
    extractor = RuleConstraintExtractor()
    for prompt in (
        "Explain binary search.",
        "Explain the history of binary search.",
        "Compare TF-IDF and Sentence Transformers.",
    ):
        assert extractor.extract(prompt) == []


@pytest.mark.parametrize(
    "prompt,expected_action",
    [
        ("Explain binary search.", "explain"),
        ("Explain the history of binary search.", "explain"),
        ("Compare TF-IDF and Sentence Transformers.", "compare"),
    ],
)
def test_ordinary_instruction_content_has_only_its_intended_requirement(prompt, expected_action):
    requirements = SpacyRequirementExtractor().extract(prompt)
    assert len(requirements) == 1
    assert requirements[0].action == expected_action


@pytest.mark.parametrize("mode", ["direct", "guided"])
def test_evaluation_request_accepts_both_optimization_modes(mode):
    request = EvaluationRequest(
        original_prompt="Original", optimized_prompt="Optimized", mode=mode, optimizer_model="model-a"
    )
    assert request.mode == mode


@pytest.mark.parametrize("mode", ["other", "DIRECT", ""])
def test_evaluation_request_rejects_invalid_mode(mode):
    with pytest.raises(ValueError):
        EvaluationRequest(
            original_prompt="Original", optimized_prompt="Optimized", mode=mode, optimizer_model="model-a"
        )


def test_evaluator_returns_typed_structured_metrics_and_metadata():
    class FixedSimilarity:
        def __init__(self, model_name):
            self.model_name = model_name

        def calculate(self, original, optimized):
            from app.evaluation.types import SemanticSimilarityMetric

            return SemanticSimilarityMetric(score=0.75)

    evaluator = PromptEvaluator(
        tokenizer=WordTokenizer(),
        requirement_extractor=FakeRequirementExtractor({"Original prompt": [requirement()], "Optimized prompt": [requirement()]}),
        constraint_extractor=FakeConstraintExtractor({"Original prompt": [constraint()], "Optimized prompt": [constraint()]}),
        similarity_factory=FixedSimilarity,
    )
    result = evaluator.evaluate(
        EvaluationRequest(
            original_prompt="Original prompt",
            optimized_prompt="Optimized prompt",
            mode="guided",
            optimizer_model="provider/model-x",
        )
    )
    data = result.model_dump(mode="json")
    assert data["evaluation_id"]
    assert data["mode"] == "guided"
    assert data["optimizer_model"] == "provider/model-x"
    assert data["tokenizer"] == "test/word"
    assert data["embedding_model"] == "all-MiniLM-L6-v2"
    assert data["metrics"]["semantic_similarity"]["score"] == 0.75
    assert data["metrics"]["requirement_preservation"]["preservation_score"] == 1.0
    assert data["metrics"]["constraint_preservation"]["preservation_score"] == 1.0
    assert data["evaluation_latency_ms"] >= 0


def test_evaluation_metrics_are_deterministic_and_inputs_are_unchanged():
    original = "Explain binary search."
    optimized = "Explain binary search clearly."
    original_before, optimized_before = original, optimized
    req = EvaluationRequest(
        original_prompt=original,
        optimized_prompt=optimized,
        mode="direct",
        optimizer_model="provider/model-a",
    )

    class FixedSimilarity:
        def __init__(self, model_name):
            pass

        def calculate(self, original_prompt, optimized_prompt):
            from app.evaluation.types import SemanticSimilarityMetric

            return SemanticSimilarityMetric(score=0.8)

    evaluator = PromptEvaluator(
        tokenizer=WordTokenizer(),
        requirement_extractor=FakeRequirementExtractor({}),
        constraint_extractor=FakeConstraintExtractor({}),
        similarity_factory=FixedSimilarity,
    )
    first = evaluator.evaluate(req)
    second = evaluator.evaluate(req)
    assert first.metrics.model_dump() == second.metrics.model_dump()
    assert original == original_before
    assert optimized == optimized_before


def test_custom_tokenizer_identifier_is_used_for_both_prompt_counts(monkeypatch):
    import app.evaluation.evaluator as evaluator_module

    tokenizer = WordTokenizer()
    calls = []

    def get_tokenizer(key):
        calls.append(key)
        return tokenizer

    class FixedSimilarity:
        def __init__(self, model_name):
            assert model_name == "test/embedding"

        def calculate(self, original, optimized):
            from app.evaluation.types import SemanticSimilarityMetric

            return SemanticSimilarityMetric(score=1.0)

    monkeypatch.setattr(evaluator_module, "get_tokenizer", get_tokenizer)
    result = PromptEvaluator(
        requirement_extractor=FakeRequirementExtractor({}),
        constraint_extractor=FakeConstraintExtractor({}),
        similarity_factory=FixedSimilarity,
    ).evaluate(
        EvaluationRequest(
            original_prompt="one two",
            optimized_prompt="one",
            mode="guided",
            optimizer_model="provider/model-b",
            tokenizer="custom-key",
            embedding_model="test/embedding",
        )
    )
    assert calls == ["custom-key"]
    assert result.tokenizer == "test/word"
    assert result.embedding_model == "test/embedding"


def test_api_returns_structured_evaluation_response(monkeypatch):
    import app.api.routes.evaluation as evaluation_route

    class FixedSimilarity:
        def __init__(self, model_name):
            self.model_name = model_name

        def calculate(self, original, optimized):
            from app.evaluation.types import SemanticSimilarityMetric

            return SemanticSimilarityMetric(score=1.0)

    monkeypatch.setattr(
        evaluation_route,
        "evaluator",
        PromptEvaluator(
            tokenizer=WordTokenizer(),
            requirement_extractor=FakeRequirementExtractor({}),
            constraint_extractor=FakeConstraintExtractor({}),
            similarity_factory=FixedSimilarity,
        ),
    )
    response = TestClient(app).post(
        "/api/evaluate",
        json={
            "original_prompt": "Explain binary search.",
            "optimized_prompt": "Explain binary search clearly.",
            "mode": "direct",
            "optimizer_model": "provider/model-a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "direct"
    assert body["optimizer_model"] == "provider/model-a"
    assert body["metrics"]["token_metrics"]["original_tokens"] == 3
    assert body["metrics"]["semantic_similarity"]["score"] == 1.0
    assert set(body) == {
        "evaluation_id", "original_prompt", "optimized_prompt", "mode", "optimizer_model",
        "tokenizer", "embedding_model", "metrics", "evaluation_latency_ms",
    }


def test_api_rejects_invalid_mode():
    response = TestClient(app).post(
        "/api/evaluate",
        json={
            "original_prompt": "Original",
            "optimized_prompt": "Optimized",
            "mode": "benchmark",
            "optimizer_model": "provider/model-a",
        },
    )
    assert response.status_code == 422


def test_constraint_matching_respects_quantity_and_format_values():
    original = [
        constraint("quantity", "Quantity constraint: 5 items requested"),
        constraint("format", "Output format: table", outputFormat="table"),
    ]
    optimized = [
        constraint("quantity", "Quantity constraint: 3 items requested"),
        constraint("format", "Output format: table", outputFormat="table"),
    ]
    extractor = FakeConstraintExtractor({"original": original, "optimized": optimized})
    result = calculate_constraint_preservation("original", "optimized", extractor)
    assert result.preserved_count == 1
    assert result.preservation_score == 0.5


def test_length_matching_respects_units_even_when_module3_maxwords_field_does_not():
    original = "Explain in under 100 words."
    optimized = "Explain in under 100 characters."
    result = calculate_constraint_preservation(original, optimized, RuleConstraintExtractor())
    assert result.original_constraints[0].maxWords == result.optimized_constraints[0].maxWords == 100
    assert result.preserved_count == 0
    assert result.preservation_score == 0.0
