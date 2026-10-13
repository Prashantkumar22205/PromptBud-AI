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

# ---------------------------------------------------------------------------
# Topic extraction configuration & generic/reference term dictionaries
# ---------------------------------------------------------------------------

GENERIC_REFERENCE_TERMS: Set[str] = {
    "guide", "my guide", "our guide", "user", "the user", "assistant", "system",
    "situation", "stuff", "thing", "things", "above", "below", "option",
    "options", "project", "my project", "our project", "answer", "response",
    "question", "prompt", "methodology", "approach", "way", "ways", "case",
    "example", "detail", "details", "info", "information", "task", "problem",
    "solution", "help", "idea", "need", "interview", "my interview", "our interview",
    "this", "that", "these", "those", "it", "its", "them", "they", "we", "us",
    "you", "your", "i", "my", "mine", "me", "he", "him", "his", "she", "her",
    "hers", "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone", "beginner", "student", "novice", "learner", "expert"
}

PRONOUNS_DEMONSTRATIVES: Set[str] = {
    "this", "that", "these", "those", "it", "its", "them", "they", "we", "us",
    "you", "your", "i", "my", "mine", "me", "he", "him", "his", "she", "her",
    "hers", "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone"
}

DETERMINER_POSSESSIVE_REGEX = re.compile(
    r"^(a|an|the|my|our|your|his|her|its|their|this|that|these|those)\s+",
    re.IGNORECASE,
)

INSTRUCTION_VERB_PATTERNS = re.compile(
    r"\b(explain|describe|teach|discuss|analyze|compare|write|summarize|parse|optimize|understand|study|implement|build|create|evaluate|solve|generate|code)\b",
    re.IGNORECASE,
)


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
        """
        Scored NLP Topic Extraction Pipeline (Deterministic & Non-LLM):
        ---------------------------------------------------------------
        Input prompt
          │
          ▼
        1. spaCy NLP Processing (noun chunking & entity recognition)
          │
          ▼
        2. Candidate Normalization
             - Strip leading determiners & possessives ("my", "our", "the", "this")
          │
          ▼
        3. Generic & Reference Expression Filtering
             - Filter phrases consisting solely of pronouns, demonstratives, or meta-terms
               ("our guide", "the user", "this", "above", "situation", "things", etc.)
          │
          ▼
        4. Transparent Feature-Based Candidate Scoring
             - Base score: 0.50
             - Domain Term Match Boost (+0.30)
             - Instruction Verb Proximity Boost (+0.20)
             - Multi-word Noun Compound Bonus (+0.10)
             - Proper Noun / Named Entity Bonus (+0.10)
             - Generic Word Penalty (-0.25)
             - Position Penalty (-0.05 per 100 chars)
          │
          ▼
        5. Selection & Bounded Confidence Calculation
             - Select candidate with highest score (if score >= 0.40)
             - Return SemanticField(value, confidence, method="spacy_noun_phrase_scored", evidence)
             - If no candidate score >= 0.40, return None
        """
        from app.semantic.types import SemanticField
        from app.semantic.utils.patterns import DOMAIN_KEYWORDS

        nlp = _get_nlp()
        if nlp is None:
            return self._extract_topic_fallback(prompt)

        doc = nlp(prompt[:5000])

        # Collect domain terms for scoring boost
        domain_terms: Set[str] = set()
        for dom_kw_set in DOMAIN_KEYWORDS.values():
            domain_terms.update(dom_kw_set)
        domain_terms.update({
            "dsa", "tf-idf", "tfidf", "sentence transformer", "embeddings",
            "binary search", "algorithm", "intent", "classification", "c++",
            "python", "java", "sql", "prompt optimization", "optimization"
        })

        # Collect candidate spans from 3 sources:
        # Source 1: spaCy noun chunks
        # Source 2: Direct targets following instruction verbs (e.g. "explain how binary search works")
        # Source 3: Known domain terms matching in prompt text
        raw_candidates = []
        for chunk in doc.noun_chunks:
            raw_candidates.append((chunk.text.strip(), chunk.start_char))

        for m in re.finditer(r"\b(explain|describe|teach|discuss|analyze|compare|write|summarize|parse|optimize|understand|study|implement|solve|generate|code)\b(?:\s+how|\s+what|\s+why|\s+about|\s+the|\s+a|\s+an)?\s+([a-zA-Z0-9\s+#\-]+?)(?:\s+works|\s+is|\s+are|\s+does|\s+to|\s+for|\s+with|\s+in|\.|\?|,|$)", prompt, re.IGNORECASE):
            target = m.group(2).strip()
            if len(target) >= 2:
                raw_candidates.append((target, m.start(2)))

        for term in domain_terms:
            if len(term) >= 3:
                for match in re.finditer(r"\b" + re.escape(term) + r"\b", prompt, re.IGNORECASE):
                    raw_candidates.append((match.group(0), match.start()))

        candidates = []
        seen_texts: Set[str] = set()

        for raw_text, char_start in raw_candidates:
            if len(raw_text) < 2:
                continue

            trimmed_text = DETERMINER_POSSESSIVE_REGEX.sub("", raw_text).strip()
            if not trimmed_text:
                continue

            trimmed_lower = trimmed_text.lower()
            if trimmed_lower in seen_texts:
                continue
            seen_texts.add(trimmed_lower)

            words = [w for w in re.findall(r"\b[a-zA-Z0-9+#\-]+\b", trimmed_lower)]
            if not words:
                continue

            # Filter phrases that consist entirely of generic/reference words
            if all(w in GENERIC_REFERENCE_TERMS for w in words):
                continue
            if trimmed_lower in PRONOUNS_DEMONSTRATIVES:
                continue

            # Transparent feature scoring
            base_score = 0.50

            has_domain_term = any(
                kw in trimmed_lower or (len(kw) > 2 and kw in words)
                for kw in domain_terms
            )
            domain_boost = 0.30 if has_domain_term else 0.0

            verb_boost = 0.0
            prompt_prefix = prompt[:char_start].lower()
            direct_verb_match = re.search(
                r"\b(explain|describe|teach|discuss|analyze|compare|write|summarize|parse|optimize|understand|study|implement|solve|generate|code)(\s+how|\s+about|\s+the|\s+a|\s+an)?\s*$",
                prompt_prefix.strip(),
                re.IGNORECASE,
            )
            if direct_verb_match or INSTRUCTION_VERB_PATTERNS.search(prompt_prefix):
                verb_boost = 0.35 if direct_verb_match else 0.15

            compound_boost = 0.10 if len(words) >= 2 else 0.0
            
            # Check for Proper Noun / Entity in prompt slice
            entity_boost = 0.0
            slice_doc = doc.char_span(char_start, char_start + len(raw_text))
            if slice_doc and any(t.pos_ == "PROPN" or t.ent_type_ for t in slice_doc):
                entity_boost = 0.10

            # Situational/meta framing penalty (e.g. "interview", "project", "guide", "situation")
            generic_penalty = 0.0
            SITUATIONAL_TERMS = {"interview", "project", "homework", "assignment", "exam", "test", "guide", "situation", "methodology", "approach", "task"}
            if any(w in SITUATIONAL_TERMS for w in words):
                generic_penalty = 0.25
            elif any(w in GENERIC_REFERENCE_TERMS for w in words) and not has_domain_term:
                generic_penalty = 0.25

            pos_penalty = min(0.15, (char_start / 100.0) * 0.05)

            final_score = base_score + domain_boost + verb_boost + compound_boost + entity_boost - generic_penalty - pos_penalty
            final_score = max(0.0, min(1.0, final_score))

            if final_score >= 0.40:
                candidates.append({
                    "value": trimmed_lower,
                    "score": final_score,
                    "evidence": raw_text,
                    "start": char_start,
                })

        if not candidates:
            return self._extract_topic_fallback(prompt)

        best = max(candidates, key=lambda c: (c["score"], -c["start"]))
        confidence = round(min(0.95, max(0.40, best["score"])), 2)

        return SemanticField(
            value=best["value"],
            confidence=confidence,
            method="spacy_noun_phrase_scored",
            evidence=best["evidence"],
        )

    def _extract_topic_fallback(self, prompt: str):
        """Regex fallback when spaCy is unavailable or noun chunks produce no candidate."""
        from app.semantic.types import SemanticField

        m = re.search(
            r"\b(explain|describe|teach|discuss|analyze|compare|write about|summarize)\s+([a-zA-Z0-9\s+#\-]+?)(?:\s+to|\s+for|\s+with|\s+in|\.|\?|$)",
            prompt, re.I,
        )
        if m:
            raw_obj = m.group(2).strip()
            trimmed_obj = DETERMINER_POSSESSIVE_REGEX.sub("", raw_obj).strip().lower()
            words = [w for w in re.findall(r"\b[a-zA-Z0-9+#\-]+\b", trimmed_obj)]
            if trimmed_obj and words and not all(w in GENERIC_REFERENCE_TERMS for w in words):
                return SemanticField(
                    value=trimmed_obj,
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
