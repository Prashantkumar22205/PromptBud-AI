"""Typed request, response, metadata, and service error contracts for Module 4."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.semantic.types import SemanticAnalysis


OptimizationMode = Literal["direct", "guided"]


class OptimizationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Original prompt to rewrite")
    mode: OptimizationMode
    model: str = Field(..., min_length=1, description="LiteLLM model identifier")
    semantic_profile: Optional[SemanticAnalysis] = Field(
        None,
        description="Module 3 semantic analysis; required for guided mode and ignored in direct mode",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return value

    @field_validator("model")
    @classmethod
    def model_must_not_be_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model identifier cannot be empty.")
        return value

    @model_validator(mode="after")
    def guided_requires_semantic_profile(self) -> "OptimizationRequest":
        if self.mode == "guided" and self.semantic_profile is None:
            raise ValueError("semantic_profile is required when mode is 'guided'.")
        return self


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = Field(None, ge=0)
    output_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)


class OptimizationMetadata(BaseModel):
    latency_ms: int = Field(..., ge=0)
    usage: Optional[TokenUsage] = None


class OptimizationResponse(BaseModel):
    mode: OptimizationMode
    model: str
    original_prompt: str
    optimized_prompt: str
    metadata: OptimizationMetadata


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    error: ErrorDetail


class OptimizationError(Exception):
    """Safe, structured error raised by the optimization service/LLM adapter."""

    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class LLMCompletion(BaseModel):
    """Normalized result returned by a LiteLLM-compatible client."""

    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
