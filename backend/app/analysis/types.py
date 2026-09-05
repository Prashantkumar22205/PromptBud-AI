"""
analysis/types.py

Pydantic models that define the structure of every analysis result.
All fields use camelCase aliases so the JSON output matches the frontend
convention without any manual renaming in route handlers.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class TextStats(BaseModel):
    """Word-level and sentence-level text statistics."""

    characterCount: int = Field(..., description="Total number of characters including whitespace")
    wordCount: int = Field(..., description="Total number of words")
    uniqueWordCount: int = Field(..., description="Number of distinct words (case-insensitive, punctuation stripped)")
    sentenceCount: int = Field(..., description="Number of sentences detected by sentence splitter")
    lineCount: int = Field(..., description="Number of non-empty lines")
    averageWordLength: float = Field(..., description="Mean character length of all words")
    averageSentenceLength: float = Field(..., description="Mean word count per sentence")
    minSentenceLength: int = Field(..., description="Shortest sentence word count")
    maxSentenceLength: int = Field(..., description="Longest sentence word count")


class TokenStats(BaseModel):
    """Token-count result from a specific tokenizer."""

    tokenCount: int = Field(..., description="Number of tokens as measured by the tokenizer")
    tokenizer: str = Field(..., description="Identifier of the tokenizer used (e.g. 'tiktoken/cl100k_base')")


class RepeatedWord(BaseModel):
    """A single repeated meaningful word entry."""

    word: str
    count: int


class RepeatedPhrase(BaseModel):
    """A single repeated n-gram (2- or 3-word) entry."""

    phrase: str
    count: int
    ngramSize: int = Field(..., description="Number of tokens in the phrase (2 or 3)")


class RepetitionStats(BaseModel):
    """
    Repetition analysis results.

    Score formula (documented):
        Let M  = count of meaningful tokens (stopwords excluded).
        Let RW = sum of (count - 1) for every repeated meaningful word.
        Let RP = sum of (count - 1) for every repeated n-gram.
        repetitionScore = min(1.0, (RW + RP) / max(1, M))

    The score is 0 when nothing repeats and approaches 1 when almost every
    token participates in a repeated pattern.  Values above ~0.3 typically
    indicate noticeable redundancy.
    """

    repeatedWords: List[RepeatedWord] = Field(default_factory=list)
    repeatedPhrases: List[RepeatedPhrase] = Field(default_factory=list)
    repetitionScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Deterministic repetition score in [0, 1]. See formula in class docstring.",
    )


class StructureStats(BaseModel):
    """
    Lightweight structural signal detection.

    Detection is rule-based (regex + keyword patterns).  It identifies
    *signals* for common prompt components; it does NOT claim full semantic
    understanding of the prompt's intent.
    """

    hasRoleInstruction: bool = Field(..., description="Detected 'you are ...' / 'act as ...' pattern")
    hasContext: bool = Field(..., description="Detected audience / background context signal")
    hasTaskInstruction: bool = Field(..., description="Detected action verb (explain, write, generate, …)")
    hasOutputFormat: bool = Field(..., description="Detected output-format directive (bullet points, JSON, table, …)")
    hasConstraints: bool = Field(..., description="Detected explicit constraint (do not …, keep it …, limit …)")
    hasExamples: bool = Field(..., description="Detected example signal (for example, e.g., such as, …)")
    hasQuestion: bool = Field(..., description="Detected at least one interrogative sentence")
    sections: List[str] = Field(
        default_factory=list,
        description="Human-readable list of detected structural sections",
    )


class PromptAnalysis(BaseModel):
    """Top-level analysis result combining all sub-modules."""

    textStats: TextStats
    tokenStats: TokenStats
    repetition: RepetitionStats
    structure: StructureStats


# ---------------------------------------------------------------------------
# API request / response envelopes
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Validated request body for POST /api/analyze."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to analyse.  Must be a non-empty string.",
    )


class AnalyzeResponse(BaseModel):
    """Successful response envelope."""

    success: bool = True
    analysis: PromptAnalysis


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Error response envelope."""

    success: bool = False
    error: ErrorDetail
