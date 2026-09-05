"""
semantic/context/extractor.py

Context extraction from prompts.

Extracts:
  - topic          (spaCy noun phrases + TF-IDF key terms)
  - domain         (keyword dictionary matching)
  - audience       (rule/pattern matching)
  - programmingLanguage (regex pattern matching)
  - priorKnowledge (explicit statement detection)

No LLM API is called.  spaCy is used for local NLP processing.

Phase 3B extension points
--------------------------
  BaseContextExtractor abstract class is defined here.
  A future SentenceTransformerContextExtractor can subclass it for
  semantic similarity-based topic/domain extraction.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_nlp():
    """Lazy-load spaCy model (cached by spaCy internally)."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found.  "
            "Run: python -m spacy download en_core_web_sm"
        )
        return None
    except ImportError:
        logger.warning("spaCy is not installed.  Run: pip install spacy")
        return None


# ---------------------------------------------------------------------------
# Abstract interface (Phase 3B extensibility)
# ---------------------------------------------------------------------------


class BaseContextExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, prompt: str) -> "ContextResult":
        """Extract context from the prompt and return a ContextResult."""


# ---------------------------------------------------------------------------
# Phase 3A implementation
# ---------------------------------------------------------------------------


class SpacyContextExtractor(BaseContextExtractor):
    """
    Context extractor using spaCy + rule-based patterns.

    spaCy provides:
    - tokenization
    - POS tagging
    - dependency parsing
    - noun phrase chunking
    - named entity recognition

    Pattern matching provides:
    - audience detection
    - programming language detection
    - prior knowledge extraction
    - domain classification
    """

    def extract(self, prompt: str) -> dict:
        from app.semantic.types import ContextResult, SemanticField

        topic = self._extract_topic(prompt)
        domain = self._extract_domain(prompt)
        audience = self._extract_audience(prompt)
        lang = self._extract_language(prompt)
        prior = self._extract_prior_knowledge(prompt)

        return ContextResult(
            topic=topic,
            domain=domain,
            audience=audience,
            programmingLanguage=lang,
            priorKnowledge=prior,
        )

    # ------------------------------------------------------------------
    # Topic extraction
    # ------------------------------------------------------------------

    def _extract_topic(self, prompt: str):
        from app.semantic.types import SemanticField

        nlp = _get_nlp()
        if nlp is None:
            return self._extract_topic_fallback(prompt)

        doc = nlp(prompt[:5000])  # cap for performance

        # Collect noun chunks, score by position (earlier = more likely topic)
        chunks = []
        for chunk in doc.noun_chunks:
            # Filter very short or stop-word-only chunks
            text = chunk.text.strip()
            if len(text) < 3:
                continue
            # Skip chunks that are just pronouns or determiners
            if all(t.pos_ in ("PRON", "DET", "ADP") for t in chunk):
                continue
            chunks.append((text, chunk.start))

        # Prefer the first meaningful noun chunk that is not a common filler
        _FILLER = re.compile(r"^(a|an|the|this|that|it|i|you|we|they|he|she|them|us)\s*$", re.I)
        for text, _ in sorted(chunks, key=lambda x: x[1]):
            if not _FILLER.match(text):
                return SemanticField(
                    value=text.lower(),
                    confidence=0.75,
                    method="spacy_noun_phrase",
                    evidence=text,
                )

        return self._extract_topic_fallback(prompt)

    def _extract_topic_fallback(self, prompt: str):
        """Simple regex fallback when spaCy is unavailable."""
        from app.semantic.types import SemanticField

        # Try to find the object of common instruction verbs
        m = re.search(
            r"\b(explain|describe|teach|discuss|analyze|compare|write about|summarize)\s+([a-zA-Z\s]+?)(?:\s+to|\s+for|\s+with|\s+in|\.|\?|$)",
            prompt, re.I,
        )
        if m:
            return SemanticField(
                value=m.group(2).strip().lower(),
                confidence=0.55,
                method="regex_verb_object",
                evidence=m.group(0).strip(),
            )
        return None

    # ------------------------------------------------------------------
    # Domain extraction
    # ------------------------------------------------------------------

    def _extract_domain(self, prompt: str):
        from app.semantic.types import SemanticField
        from app.semantic.utils.patterns import DOMAIN_KEYWORDS

        prompt_lower = prompt.lower()
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in prompt_lower)
            if hits:
                scores[domain] = hits

        if not scores:
            return None

        best_domain = max(scores, key=lambda d: scores[d])
        total_hits = scores[best_domain]
        # Normalise confidence: 1 hit → ~0.6, 3+ hits → ~0.9
        confidence = min(0.95, 0.55 + 0.12 * total_hits)

        return SemanticField(
            value=best_domain,
            confidence=round(confidence, 2),
            method="domain_dictionary",
            evidence=None,
        )

    # ------------------------------------------------------------------
    # Audience extraction
    # ------------------------------------------------------------------

    def _extract_audience(self, prompt: str):
        from app.semantic.types import SemanticField
        from app.semantic.utils.patterns import AUDIENCE_PATTERNS

        for label, pattern, confidence in AUDIENCE_PATTERNS:
            m = pattern.search(prompt)
            if m:
                return SemanticField(
                    value=label,
                    confidence=confidence,
                    method="rule",
                    evidence=m.group(0).strip(),
                )
        return None

    # ------------------------------------------------------------------
    # Programming language extraction
    # ------------------------------------------------------------------

    def _extract_language(self, prompt: str):
        from app.semantic.types import SemanticField
        from app.semantic.utils.patterns import LANGUAGE_PATTERNS

        for lang, pattern in LANGUAGE_PATTERNS:
            m = pattern.search(prompt)
            if m:
                return SemanticField(
                    value=lang,
                    confidence=0.97,
                    method="rule",
                    evidence=m.group(0).strip(),
                )
        return None

    # ------------------------------------------------------------------
    # Prior knowledge extraction
    # ------------------------------------------------------------------

    def _extract_prior_knowledge(self, prompt: str) -> list:
        from app.semantic.types import SemanticField
        from app.semantic.utils.patterns import PRIOR_KNOWLEDGE_PATTERNS

        results = []
        for kind, pattern, in [(k, p) for k, p, in PRIOR_KNOWLEDGE_PATTERNS]:
            m = pattern.search(prompt)
            if m:
                results.append(SemanticField(
                    value=kind,
                    confidence=0.85,
                    method="rule",
                    evidence=m.group(0).strip(),
                ))

        return results


def get_context_extractor() -> BaseContextExtractor:
    """Return the Phase 3A context extractor."""
    return SpacyContextExtractor()
