"""Independent Module 5 evaluation orchestration; never calls Modules 3 or 4 APIs."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from app.analysis.tokenizer import BaseTokenizer, get_tokenizer
from app.core.config import settings
from app.evaluation.metrics.constraint_metrics import calculate_constraint_preservation
from app.evaluation.metrics.requirement_metrics import calculate_requirement_preservation
from app.evaluation.metrics.semantic_metrics import SentenceTransformerSimilarity
from app.evaluation.metrics.token_metrics import calculate_token_metrics
from app.evaluation.types import (
    EvaluationError,
    EvaluationMetrics,
    EvaluationRequest,
    EvaluationResponse,
)
from app.semantic.constraints.extractor import BaseConstraintExtractor, RuleConstraintExtractor
from app.semantic.requirements.extractor import BaseRequirementExtractor, SpacyRequirementExtractor


class PromptEvaluator:
    """Compute independent metrics for two caller-supplied prompt strings."""

    def __init__(
        self,
        *,
        tokenizer: BaseTokenizer | None = None,
        requirement_extractor: BaseRequirementExtractor | None = None,
        constraint_extractor: BaseConstraintExtractor | None = None,
        similarity_factory=None,
    ):
        self._tokenizer = tokenizer
        self._requirement_extractor = requirement_extractor or SpacyRequirementExtractor()
        self._constraint_extractor = constraint_extractor or RuleConstraintExtractor()
        self._similarity_factory = similarity_factory or SentenceTransformerSimilarity

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        started = perf_counter()
        tokenizer_name = request.tokenizer or settings.default_tokenizer
        embedding_model = request.embedding_model or settings.evaluation_embedding_model

        try:
            tokenizer = self._tokenizer or get_tokenizer(tokenizer_name)
        except ValueError as exc:
            raise EvaluationError("UNKNOWN_TOKENIZER", str(exc), 422) from exc
        except Exception as exc:
            raise EvaluationError("TOKENIZER_UNAVAILABLE", "The configured tokenizer could not be loaded.") from exc

        try:
            token_metrics = calculate_token_metrics(
                request.original_prompt, request.optimized_prompt, tokenizer
            )
            semantic_similarity = self._similarity_factory(embedding_model).calculate(
                request.original_prompt, request.optimized_prompt
            )
            requirement_preservation = calculate_requirement_preservation(
                request.original_prompt,
                request.optimized_prompt,
                self._requirement_extractor,
                self._constraint_extractor,
            )
            constraint_preservation = calculate_constraint_preservation(
                request.original_prompt, request.optimized_prompt, self._constraint_extractor
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError("METRIC_FAILED", "A prompt evaluation metric could not be computed.") from exc

        return EvaluationResponse(
            evaluation_id=uuid4(),
            original_prompt=request.original_prompt,
            optimized_prompt=request.optimized_prompt,
            mode=request.mode,
            optimizer_model=request.optimizer_model,
            tokenizer=tokenizer.name,
            embedding_model=embedding_model,
            metrics=EvaluationMetrics(
                token_metrics=token_metrics,
                semantic_similarity=semantic_similarity,
                requirement_preservation=requirement_preservation,
                constraint_preservation=constraint_preservation,
            ),
            evaluation_latency_ms=max(0, int((perf_counter() - started) * 1000)),
        )
