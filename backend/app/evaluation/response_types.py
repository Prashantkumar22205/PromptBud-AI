"""Typed contracts for downstream response evaluation (Module 5B)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


ResponseVariant = Literal["original", "direct", "guided"]
CheckStatus = Literal["passed", "failed", "not_evaluated"]
CostStatus = Literal["provided", "estimated", "unavailable"]


class ResponseEvaluationRequest(BaseModel):
    original_prompt: str = Field(..., min_length=1)
    evaluation_prompt: str | None = Field(None, min_length=1)
    response: str = Field(..., min_length=1)
    variant: ResponseVariant
    target_model: str = Field(..., min_length=1)
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    total_tokens: int | None = Field(None, ge=0)
    latency_ms: float | None = Field(None, ge=0)
    estimated_cost: float | None = Field(None, ge=0)
    generation_temperature: float | None = Field(None, ge=0, le=2)
    max_output_tokens: int | None = Field(None, gt=0)

    @field_validator("original_prompt", "response", "target_model")
    @classmethod
    def values_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be empty or whitespace only.")
        return value

    @field_validator("evaluation_prompt")
    @classmethod
    def evaluation_prompt_must_not_be_whitespace(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Value cannot be empty or whitespace only.")
        return value

    @field_validator("target_model")
    @classmethod
    def strip_target_model(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def token_total_must_be_consistent(self) -> "ResponseEvaluationRequest":
        if self.evaluation_prompt is None:
            self.evaluation_prompt = self.original_prompt
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens != self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens must equal input_tokens + output_tokens when all are provided.")
        return self

    def resolved_total_tokens(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class ScoreWithReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: StrictInt = Field(..., ge=1, le=5)
    reason: str = Field(..., min_length=1, max_length=1000)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A concise reason is required for every score.")
        return value.strip()


class ResponseQualityScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: ScoreWithReason
    completeness: ScoreWithReason
    correctness: ScoreWithReason
    instruction_following: ScoreWithReason


class DeterministicInstructionCheck(BaseModel):
    constraint_type: str
    status: CheckStatus
    reason: str
    expected: str | None = None
    observed: str | None = None


class TokenUsageMetric(BaseModel):
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    total_tokens: int | None = Field(None, ge=0)


class CostMetric(BaseModel):
    estimated_cost: float | None = Field(None, ge=0)
    status: CostStatus
    reason_code: str | None = None
    reason: str


class ResponseEvaluationMetrics(BaseModel):
    token_usage: TokenUsageMetric
    target_model_latency_ms: float | None = Field(None, ge=0)
    cost: CostMetric
    deterministic_instruction_checks: list[DeterministicInstructionCheck]
    response_quality: ResponseQualityScores


class ResponseEvaluatorInfo(BaseModel):
    method: Literal["independent_llm_judge"] = "independent_llm_judge"
    model: str
    latency_ms: float = Field(..., ge=0)


class ResponseEvaluationResult(BaseModel):
    evaluation_id: UUID
    original_prompt: str
    evaluation_prompt: str
    response: str
    variant: ResponseVariant
    target_model: str
    generation_temperature: float | None = Field(None, ge=0, le=2)
    max_output_tokens: int | None = Field(None, gt=0)
    metrics: ResponseEvaluationMetrics
    evaluator: ResponseEvaluatorInfo


class ResponseEvaluationError(Exception):
    """Safe structured error for evaluator/configuration failures."""

    def __init__(self, code: str, message: str, status_code: int = 503):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
