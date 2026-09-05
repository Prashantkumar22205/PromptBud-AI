"""
semantic/types.py

Pydantic models for Module 3 — Semantic Analysis.

All API-facing fields use camelCase to match the existing project convention.
Python-internal code uses snake_case.

Important notes
---------------
- confidence values from rule-based systems are heuristic estimates.
- confidence values from the ML classifier are model probability scores,
  not guaranteed correctness measures.
- method field documents HOW the value was produced for explainability.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared field model
# ---------------------------------------------------------------------------


class SemanticField(BaseModel):
    """
    A single extracted semantic field with provenance metadata.
    Used for context sub-fields that have a scalar value.
    """

    value: str = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Heuristic or model confidence in [0, 1]")
    method: str = Field(..., description="Detection method (e.g. 'rule', 'spacy_noun_phrase', 'domain_dictionary')")
    evidence: Optional[str] = Field(None, description="Source text span used to produce this extraction")


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class IntentResult(BaseModel):
    """
    Intent classification result.

    The classifier uses TF-IDF + Logistic Regression trained on a small
    development seed dataset (data/intent_samples.json).

    IMPORTANT: Classifier probability (confidence) is a model estimate —
    not a guarantee of correctness.  The seed dataset is a development
    baseline, NOT the final research evaluation dataset.
    """

    primaryIntent: str = Field(..., description="Predicted primary intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier probability for the predicted class")
    method: str = Field(default="tfidf_logistic_regression", description="Classification method identifier")
    secondaryIntents: Optional[List[str]] = Field(None, description="Top secondary intent labels if any")
    allScores: Optional[dict] = Field(None, description="All class probabilities for transparency (optional)")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class ContextResult(BaseModel):
    """Extracted contextual metadata about the prompt."""

    topic: Optional[SemanticField] = Field(None, description="Main topic or key concept")
    domain: Optional[SemanticField] = Field(None, description="Detected knowledge domain")
    audience: Optional[SemanticField] = Field(None, description="Target audience")
    programmingLanguage: Optional[SemanticField] = Field(None, description="Programming language if detected")
    priorKnowledge: List[SemanticField] = Field(default_factory=list, description="Explicitly stated prior knowledge assumptions")


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    """
    A single detected requirement — something the optimizer must preserve.

    Requirements represent the user's explicit intentions that must not be
    removed or altered during prompt optimization.
    """

    action: str = Field(..., description="Action verb (e.g. 'explain', 'provide', 'generate')")
    object: Optional[str] = Field(None, description="Object of the action (what to act on)")
    mandatory: bool = Field(default=True, description="Whether this requirement appears mandatory")
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str = Field(..., description="Detection method")
    evidence: Optional[str] = Field(None, description="Source text that produced this requirement")


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class Constraint(BaseModel):
    """
    A single detected constraint — a restriction the optimizer must respect.
    """

    type: str = Field(..., description="Constraint category: length, format, style, language, technology, forbidden, required, scope, audience")
    description: str = Field(..., description="Constraint description")
    maxWords: Optional[int] = Field(None, description="Maximum word limit if length constraint")
    minWords: Optional[int] = Field(None, description="Minimum word limit if length constraint")
    outputFormat: Optional[str] = Field(None, description="Requested output format if format constraint")
    targetLanguage: Optional[str] = Field(None, description="Target natural language if language constraint")
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str = Field(..., description="Detection method")
    evidence: Optional[str] = Field(None, description="Source text that produced this constraint")


# ---------------------------------------------------------------------------
# Ambiguities
# ---------------------------------------------------------------------------


class Ambiguity(BaseModel):
    """
    A detected ambiguity or conflict in the prompt.

    Phase 3A uses rule-based detection only.
    Phase 3B will add NLI-based contradiction detection.
    """

    type: str = Field(
        ...,
        description=(
            "Ambiguity type: conflicting_constraints, underspecified_object, "
            "missing_context, referential_ambiguity, other"
        ),
    )
    description: str = Field(..., description="Human-readable description of the ambiguity")
    severity: str = Field(..., description="low | medium | high")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    method: str = Field(default="rule", description="Detection method")
    evidence: Optional[str] = Field(None, description="Text span that triggered this ambiguity")
    suggestedResolution: Optional[str] = Field(None, description="Suggested resolution for the user")


# ---------------------------------------------------------------------------
# Top-level semantic analysis
# ---------------------------------------------------------------------------


class SemanticAnalysis(BaseModel):
    """
    Complete semantic analysis of a prompt.

    Produced by Module 3 — Semantic Analysis (Phase 3A).
    No commercial LLM API is called to produce this result.
    """

    intent: IntentResult
    context: ContextResult
    requirements: List[Requirement] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API envelopes
# ---------------------------------------------------------------------------


class SemanticAnalyzeRequest(BaseModel):
    """Request body for POST /api/semantic-analyze."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to analyse semantically. Must be a non-empty string.",
    )


class SemanticAnalyzeResponse(BaseModel):
    """Successful response envelope for POST /api/semantic-analyze."""

    success: bool = True
    analysis: SemanticAnalysis

