"""Typed request, metric, result, and error contracts for Module 5."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.semantic.types import Constraint, Requirement


EvaluationMode = Literal["direct", "guided"]


class EvaluationRequest(BaseModel):
    original_prompt: str = Field(..., min_length=1)
    optimized_prompt: str = Field(..., min_length=1)
    mode: EvaluationMode
    optimizer_model: str = Field(..., min_length=1)
    tokenizer: str | None = Field(None, description="Tokenizer registry key; defaults to configured tokenizer")
    embedding_model: str | None = Field(None, description="SentenceTransformer identifier; defaults to configured model")

    @field_validator("original_prompt", "optimized_prompt")
    @classmethod
    def prompt_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return value

    @field_validator("optimizer_model", "tokenizer", "embedding_model")
    @classmethod
    def identifiers_must_not_be_whitespace(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("Identifier cannot be empty or whitespace only.")
        return value


class TokenMetrics(BaseModel):
    original_tokens: int = Field(..., ge=0)
    optimized_tokens: int = Field(..., ge=0)
    token_delta: int
    token_reduction_percent: float


class SemanticSimilarityMetric(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)


class RequirementPreservationMetric(BaseModel):
    original_requirements: list[Requirement]
    optimized_requirements: list[Requirement]
    preserved_count: int = Field(..., ge=0)
    original_count: int = Field(..., ge=0)
    preservation_score: float = Field(..., ge=0.0, le=1.0)


class ConstraintPreservationMetric(BaseModel):
    original_constraints: list[Constraint]
    optimized_constraints: list[Constraint]
    preserved_count: int = Field(..., ge=0)
    original_count: int = Field(..., ge=0)
    preservation_score: float = Field(..., ge=0.0, le=1.0)


class EvaluationMetrics(BaseModel):
    token_metrics: TokenMetrics
    semantic_similarity: SemanticSimilarityMetric
    requirement_preservation: RequirementPreservationMetric
    constraint_preservation: ConstraintPreservationMetric


class EvaluationResponse(BaseModel):
    evaluation_id: UUID
    original_prompt: str
    optimized_prompt: str
    mode: EvaluationMode
    optimizer_model: str
    tokenizer: str
    embedding_model: str
    metrics: EvaluationMetrics
    evaluation_latency_ms: int = Field(..., ge=0)


class EvaluationError(Exception):
    """Safe error surfaced by the evaluation API."""

    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
