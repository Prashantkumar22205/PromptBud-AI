"""SentenceTransformer cosine similarity; no LLM judge is used."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from app.evaluation.types import EvaluationError, SemanticSimilarityMetric


@lru_cache(maxsize=4)
def _load_sentence_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EvaluationError(
            "EMBEDDING_DEPENDENCY_UNAVAILABLE",
            "sentence-transformers is required for semantic similarity.",
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EvaluationError(
            "EMBEDDING_MODEL_UNAVAILABLE",
            "The configured semantic embedding model could not be loaded.",
        ) from exc


class SentenceTransformerSimilarity:
    def __init__(self, model_name: str, model: Any | None = None):
        self.model_name = model_name
        self._model = model

    def calculate(self, original_prompt: str, optimized_prompt: str) -> SemanticSimilarityMetric:
        model = self._model or _load_sentence_model(self.model_name)
        try:
            vectors = np.asarray(
                model.encode(
                    [original_prompt, optimized_prompt],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=float,
            )
        except Exception as exc:
            raise EvaluationError(
                "EMBEDDING_FAILED", "The embedding model could not encode the supplied prompts."
            ) from exc

        if vectors.ndim != 2 or vectors.shape[0] != 2 or not np.isfinite(vectors).all():
            raise EvaluationError("MALFORMED_EMBEDDING", "The embedding model returned invalid vectors.")

        norm_product = float(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        if norm_product == 0.0:
            cosine = 1.0 if np.array_equal(vectors[0], vectors[1]) else 0.0
        else:
            cosine = float(np.dot(vectors[0], vectors[1]) / norm_product)
        cosine = min(1.0, max(-1.0, cosine))
        # Map cosine [-1, 1] to the requested normalized similarity range [0, 1].
        return SemanticSimilarityMetric(score=(cosine + 1.0) / 2.0)
