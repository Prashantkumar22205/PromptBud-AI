"""
semantic/intent/classifier.py

ML-based intent classification supporting:
  E1: TF-IDF + Logistic Regression (baseline: TfidfLogisticIntentClassifier)
  E2: Sentence Transformer Embeddings + Logistic Regression (SemanticIntentClassifier)

Architecture
------------
  BaseIntentClassifier   ← abstract interface
        ├── TfidfLogisticIntentClassifier (E1 baseline)
        └── SemanticIntentClassifier      (E2 sentence-transformer embedding model)

Both classifiers return full probability distributions in `allScores` for auditability
and research evaluation transparency.
"""

from __future__ import annotations

import abc
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.09   # provisional threshold — below this, fallback to "other"
FALLBACK_LABEL = "other"
E1_METHOD_ID = "tfidf_logistic_regression"
E2_METHOD_ID = "sentence_transformer_logistic_regression"
E3_METHOD_ID = "hybrid_tfidf_semantic"


# ---------------------------------------------------------------------------
# Abstract interface — enables swappable research classifiers
# ---------------------------------------------------------------------------


class BaseIntentClassifier(abc.ABC):
    """
    Interface that all intent classifier implementations must satisfy.
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

    def classify(self, text: str):
        """
        Classify intent and return structured IntentResult with allScores.
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
            allScores=all_scores,
        )


# ---------------------------------------------------------------------------
# E1 Implementation: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------


class TfidfLogisticIntentClassifier(BaseIntentClassifier):
    """
    E1 Baseline: Intent classifier using TF-IDF vectorization + Logistic Regression.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self._threshold = confidence_threshold
        self._pipeline = None
        self._classes: List[str] = []
        self._is_fitted = False

    @property
    def method(self) -> str:
        return E1_METHOD_ID

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfLogisticIntentClassifier":
        """Train the pipeline on training data."""
        if not texts:
            logger.warning("No training data provided; E1 classifier will return '%s'.", FALLBACK_LABEL)
            return self

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except ImportError as exc:
            raise RuntimeError("scikit-learn is not installed. Run: pip install scikit-learn") from exc

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
        logger.info("E1 TF-IDF classifier trained on %d samples, %d classes.", len(texts), len(self._classes))
        return self

    def predict(self, text: str) -> Tuple[str, float, Optional[Dict[str, float]]]:
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
                return FALLBACK_LABEL, confidence, all_scores

            return label, confidence, all_scores

        except Exception:
            logger.exception("E1 intent prediction failed; returning fallback.")
            return FALLBACK_LABEL, 0.0, None


# Alias for backward compatibility with existing tests
TfidfLRClassifier = TfidfLogisticIntentClassifier


# ---------------------------------------------------------------------------
# E2 Implementation: Sentence Transformer + Logistic Regression
# ---------------------------------------------------------------------------


class SemanticIntentClassifier(BaseIntentClassifier):
    """
    E2 Improved Classifier: Dense sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_name = model_name
        self._threshold = confidence_threshold
        self._model = None
        self._clf = None
        self._classes: List[str] = []
        self._is_fitted = False

    @property
    def method(self) -> str:
        return E2_METHOD_ID

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers") from exc
        return self._model

    def fit(self, texts: List[str], labels: List[str]) -> "SemanticIntentClassifier":
        """Encode training texts into dense embeddings and fit LogisticRegression."""
        if not texts:
            logger.warning("No training data provided; E2 classifier will return '%s'.", FALLBACK_LABEL)
            return self

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError("scikit-learn is not installed. Run: pip install scikit-learn") from exc

        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        self._clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        self._clf.fit(embeddings, labels)
        self._classes = list(self._clf.classes_)
        self._is_fitted = True
        logger.info("E2 Semantic classifier trained on %d samples, %d classes.", len(texts), len(self._classes))
        return self

    def predict(self, text: str) -> Tuple[str, float, Optional[Dict[str, float]]]:
        if not self._is_fitted or self._clf is None:
            return FALLBACK_LABEL, 0.0, None

        text = text.strip()
        if not text:
            return FALLBACK_LABEL, 0.0, None

        try:
            model = self._load_model()
            embedding = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            proba = self._clf.predict_proba(embedding)[0]

            top_idx = int(proba.argmax())
            label = self._classes[top_idx]
            confidence = float(round(proba[top_idx], 4))

            all_scores = {cls: round(float(p), 4) for cls, p in zip(self._classes, proba)}

            if confidence < self._threshold:
                return FALLBACK_LABEL, confidence, all_scores

            return label, confidence, all_scores

        except Exception:
            logger.exception("E2 intent prediction failed; returning fallback.")
            return FALLBACK_LABEL, 0.0, None


# ---------------------------------------------------------------------------
# E3 Implementation: Hybrid Intent Classifier (Probability-level Fusion)
# ---------------------------------------------------------------------------


class HybridIntentClassifier(BaseIntentClassifier):
    """
    E3 Hybrid Intent Classifier: Probability-level fusion of E1 (TF-IDF) and E2 (Semantic).

    Formula:
      P_hybrid(intent) = alpha * P_E1(intent) + (1 - alpha) * P_E2(intent)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        e1_clf: Optional[TfidfLogisticIntentClassifier] = None,
        e2_clf: Optional[SemanticIntentClassifier] = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0.0, 1.0], got {alpha}")
        self.alpha = float(alpha)
        self._threshold = confidence_threshold
        self._e1 = e1_clf if e1_clf is not None else TfidfLogisticIntentClassifier(confidence_threshold=0.0)
        self._e2 = e2_clf if e2_clf is not None else SemanticIntentClassifier(confidence_threshold=0.0)
        self._is_fitted = self._e1._is_fitted and self._e2._is_fitted

    @property
    def method(self) -> str:
        return E3_METHOD_ID

    def fit(self, texts: List[str], labels: List[str]) -> "HybridIntentClassifier":
        if not self._e1._is_fitted:
            self._e1.fit(texts, labels)
        if not self._e2._is_fitted:
            self._e2.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict(self, text: str) -> Tuple[str, float, Optional[Dict[str, float]]]:
        if not self._is_fitted:
            return FALLBACK_LABEL, 0.0, None

        text = text.strip()
        if not text:
            return FALLBACK_LABEL, 0.0, None

        _, _, scores_e1 = self._e1.predict(text)
        _, _, scores_e2 = self._e2.predict(text)

        if not scores_e1 or not scores_e2:
            return FALLBACK_LABEL, 0.0, None

        all_classes = sorted(list(set(scores_e1.keys()) | set(scores_e2.keys())))
        hybrid_scores = {}
        for cls in all_classes:
            p1 = scores_e1.get(cls, 0.0)
            p2 = scores_e2.get(cls, 0.0)
            p_hybrid = self.alpha * p1 + (1.0 - self.alpha) * p2
            hybrid_scores[cls] = round(float(p_hybrid), 4)

        top_cls = max(hybrid_scores, key=hybrid_scores.get)
        confidence = float(hybrid_scores[top_cls])

        if confidence < self._threshold:
            return FALLBACK_LABEL, confidence, hybrid_scores

        return top_cls, confidence, hybrid_scores


# ---------------------------------------------------------------------------
# Global singletons & Factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_tfidf_classifier() -> TfidfLogisticIntentClassifier:
    """Build and train E1 TF-IDF classifier on V2 dev dataset."""
    from .training_data import load_training_data
    texts, labels = load_training_data()
    clf = TfidfLogisticIntentClassifier()
    clf.fit(texts, labels)
    return clf


@lru_cache(maxsize=1)
def _build_semantic_classifier() -> SemanticIntentClassifier:
    """Build and train E2 Semantic classifier on V2 dev dataset."""
    from .training_data import load_training_data
    texts, labels = load_training_data()
    clf = SemanticIntentClassifier()
    clf.fit(texts, labels)
    return clf


@lru_cache(maxsize=1)
def _build_hybrid_classifier(alpha: float = 0.5) -> HybridIntentClassifier:
    """Build and train E3 Hybrid classifier on V2 dev dataset."""
    from .training_data import load_training_data
    texts, labels = load_training_data()
    clf_e1 = _build_tfidf_classifier()
    clf_e2 = _build_semantic_classifier()
    clf = HybridIntentClassifier(alpha=alpha, e1_clf=clf_e1, e2_clf=clf_e2)
    clf.fit(texts, labels)
    return clf


def get_intent_classifier(classifier_type: Optional[str] = None) -> BaseIntentClassifier:
    """
    Return the configured intent classifier instance.

    Parameters
    ----------
    classifier_type : str, optional
        'tfidf' for E1 baseline, 'semantic' for E2 sentence transformer model, or 'hybrid' for E3 fusion.
        If None, reads settings.intent_classifier_type from config (defaults to 'tfidf').
    """
    from app.core.config import settings

    target_type = (classifier_type or getattr(settings, "intent_classifier_type", "tfidf")).lower()

    if target_type == "semantic":
        return _build_semantic_classifier()
    elif target_type == "hybrid":
        alpha = getattr(settings, "intent_hybrid_alpha", 0.5)
        return _build_hybrid_classifier(alpha=alpha)
    else:
        return _build_tfidf_classifier()
