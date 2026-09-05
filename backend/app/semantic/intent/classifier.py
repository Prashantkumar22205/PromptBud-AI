"""
semantic/intent/classifier.py

ML-based intent classification using TF-IDF + Logistic Regression.

Architecture
------------
  BaseIntentClassifier   ← abstract interface (for Phase 3B alternatives)
        └── TfidfLRClassifier   ← Phase 3A concrete implementation

Phase 3B can add:
  - TransformerIntentClassifier  (sentence-transformers zero-shot)
  - LLMFallbackClassifier        (optional LLM API for low-confidence cases)

The classifier is trained once on startup from the seed dataset and cached.
All predictions are fully local — no external API is called.

IMPORTANT CAVEATS
-----------------
- The seed dataset (data/intent_samples.json) is a development baseline.
  It is NOT the final research evaluation dataset.
- predict_proba() returns model confidence, not guaranteed correctness.
- Confidence threshold below which 'other' is returned is configurable.
"""

from __future__ import annotations

import abc
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.09   # configurable — below this, fallback to "other"
FALLBACK_LABEL = "other"
METHOD_ID = "tfidf_logistic_regression"


# ---------------------------------------------------------------------------
# Abstract interface — enables Phase 3B swappable classifiers
# ---------------------------------------------------------------------------


class BaseIntentClassifier(abc.ABC):
    """
    Interface that all intent classifier implementations must satisfy.

    Phase 3B implementations (Transformers, LLM fallback) should subclass
    this and override predict() and fit().
    """

    @property
    @abc.abstractmethod
    def method(self) -> str:
        """Human-readable method identifier for provenance tracking."""

    @abc.abstractmethod
    def fit(self, texts: List[str], labels: List[str]) -> "BaseIntentClassifier":
        """Train the classifier on the given texts and labels."""

    @abc.abstractmethod
    def predict(self, text: str) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Predict intent for a single text.

        Returns
        -------
        label : str
            Predicted intent label.
        confidence : float
            Model confidence for the predicted class in [0, 1].
        all_scores : dict or None
            All class probabilities for transparency.
        """

    def classify(self, text: str) -> IntentResult:
        """
        Classify intent and return structured IntentResult.
        """
        from app.semantic.types import IntentResult

        label, confidence, all_scores = self.predict(text)
        secondary: List[str] = []
        if all_scores:
            # Sort remaining classes by confidence descending
            sorted_scores = sorted(
                [(cls, score) for cls, score in all_scores.items() if cls != label],
                key=lambda item: item[1],
                reverse=True,
            )
            # Include secondary intents with score >= 0.15
            secondary = [cls for cls, score in sorted_scores if score >= 0.15][:3]

        return IntentResult(
            primaryIntent=label,
            confidence=confidence,
            method=self.method,
            secondaryIntents=secondary if secondary else None,
        )


# ---------------------------------------------------------------------------
# Phase 3A implementation: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------


class TfidfLRClassifier(BaseIntentClassifier):
    """
    Intent classifier using TF-IDF vectorization + Logistic Regression.

    This is the Phase 3A implementation.  It trains fully locally using
    scikit-learn — no external API is called.

    The pipeline is:
        raw text
            → TfidfVectorizer (n-gram range 1-2, sublinear TF)
            → LogisticRegression (max_iter=1000, C=1.0)
            → predicted label + predict_proba() scores
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self._threshold = confidence_threshold
        self._pipeline = None
        self._classes: List[str] = []
        self._is_fitted = False

    @property
    def method(self) -> str:
        return METHOD_ID

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfLRClassifier":
        """Train the pipeline on seed data."""
        if not texts:
            logger.warning("No training data provided; classifier will always return '%s'.", FALLBACK_LABEL)
            return self

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError as exc:
            raise RuntimeError("scikit-learn is not installed.  Run: pip install scikit-learn") from exc

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_features=10_000,
                strip_accents="unicode",
                analyzer="word",
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),

        ])

        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.classes_)
        self._is_fitted = True
        logger.info("Intent classifier trained on %d samples, %d classes.", len(texts), len(self._classes))
        return self

    def predict(self, text: str) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Predict intent for a single text string.

        Returns
        -------
        label, confidence, all_scores
        """
        if not self._is_fitted or self._pipeline is None:
            return FALLBACK_LABEL, 0.0, None

        text = text.strip()
        if not text:
            return FALLBACK_LABEL, 0.0, None

        try:
            proba = self._pipeline.predict_proba([text])[0]
            top_idx = int(proba.argmax())
            label = self._classes[top_idx]
            confidence = float(round(proba[top_idx], 4))

            all_scores = {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)}

            if confidence < self._threshold:
                logger.debug(
                    "Intent confidence %.3f below threshold %.3f; returning fallback '%s'.",
                    confidence, self._threshold, FALLBACK_LABEL,
                )
                return FALLBACK_LABEL, confidence, all_scores

            return label, confidence, all_scores

        except Exception:
            logger.exception("Intent classification failed; returning fallback.")
            return FALLBACK_LABEL, 0.0, None


# ---------------------------------------------------------------------------
# Global singleton — trained once at startup
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_classifier() -> TfidfLRClassifier:
    """Build and train the classifier; cached as a singleton."""
    from .training_data import load_training_data

    texts, labels = load_training_data()
    clf = TfidfLRClassifier()
    clf.fit(texts, labels)
    return clf


def get_intent_classifier() -> BaseIntentClassifier:
    """Return the singleton intent classifier instance."""
    return _build_classifier()
