"""
semantic/requirements/extractor.py

Requirement Extraction Architecture & Implementation (Module 3 — Non-LLM)

Pipeline Architecture
---------------------
Prompt
  │
  ▼
spaCy Processing (en_core_web_sm dependency parsing, POS tagging, sents)
  │
  ▼
Action Verb Identification (iterating verbs across ROOT, conj, xcomp, ccomp)
  │
  ▼
Dependency/Object Extraction (dobj, attr, pobj, clausal complements)
  │
  ▼
Coordinated-Clause & Question Detection (multi-verb conj & Wh- question rules)
  │
  ▼
Invalid-Object Filtering (rejecting standalone pronouns like 'he', 'it', 'this', fillers)
  │
  ▼
Requirement Candidate Scoring & Deduplication (bounded confidence C ∈ [0.60, 0.98])
  │
  ▼
Requirement List (Requirement objects with action, object, mandatory, confidence, evidence)

Design Characteristics
----------------------
1. Non-LLM: Uses deterministic NLP dependency trees and rule-based clausal parsing.
   Requires zero external generative LLM APIs (OpenAI, Gemini, Anthropic, or local LLMs).
2. Deterministic & Reproducible: Returns exact, identical results across runs for given input.
3. Interpretable: Every extracted requirement retains exact provenance ('evidence' span
   and 'method' identifier: 'spacy_dependency_rule', 'question_rule', 'clause_rule', 'regex_pattern').

Known Limitations
-----------------
1. Complex Coreference: Pronouns referring to antecedent entities in earlier sentences
   (e.g. "Do it again" where 'it' refers to a prior prompt context) are intentionally
   rejected as requirement objects to avoid returning invalid pronouns; reference resolution
   is left to downstream ambiguity engines.
2. Highly Ungrammatical Prompts: Syntactically broken sentences may result in spaCy misidentifying
   head verbs or objects; fallback regex and sentence scanning mitigate common patterns.
3. Implicit Requirements: Non-syntactically expressed requirements (e.g. implied sub-tasks not
   governed by an explicit action verb) are out of scope for pure rule-based dependency parsing.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Minimum confidence for a verb-object pair to be reported
MIN_CONFIDENCE = 0.60

# Standalone pronouns, determiners, and generic fillers rejected as objects
PRONOUNS_AND_FILLERS: Set[str] = {
    "he", "she", "it", "this", "that", "these", "those", "they", "we", "you",
    "i", "me", "us", "him", "her", "them", "who", "whom", "which", "what",
    "someone", "anyone", "everyone", "something", "anything", "everything",
    "thing", "things", "stuff", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "themselves", "a", "an", "the", "there", "here"
}

# Action verb normalization mapping
ACTION_NORM_MAP = {
    "tell": "explain",
    "give": "provide",
    "show": "show",
    "walk": "explain",
    "use": "provide",
    "do": "conduct",
}

# Markers indicating optional rather than mandatory requirements
OPTIONAL_MARKERS = {
    "optional", "optionally", "if possible", "if you can", "could you", "perhaps",
    "if desired", "if needed", "if applicable", "if you want"
}


def _get_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        return None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseRequirementExtractor(abc.ABC):
    @abc.abstractmethod
    def extract(self, prompt: str) -> list:
        """Return a list of Requirement objects."""


# ---------------------------------------------------------------------------
# Improved SpacyRequirementExtractor Implementation
# ---------------------------------------------------------------------------


class SpacyRequirementExtractor(BaseRequirementExtractor):
    """
    Extracts requirements using spaCy dependency parsing, question clause
    analysis, and regex patterns.
    """

    def extract(self, prompt: str) -> list:
        from app.semantic.types import Requirement

        results: List[Requirement] = []
        seen_keys: set = set()

        nlp = _get_nlp()
        if nlp:
            # Strategy 1: spaCy dependency and multi-verb clausal parsing
            results.extend(self._extract_via_spacy(prompt, nlp, seen_keys))

            # Strategy 2: Question-style & embedded clause requirements
            results.extend(self._extract_question_requirements(prompt, nlp, seen_keys))

        # Strategy 3: Explicit regex pattern rules (supplementary)
        results.extend(self._extract_via_patterns(prompt, seen_keys))

        # Strategy 4: Fallback sentence verb scan
        if not results:
            results.extend(self._extract_via_sentence_scan(prompt, seen_keys))

        return results

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _is_valid_object(self, obj_text: Optional[str]) -> bool:
        """Filter clearly invalid requirement objects (pronouns, fillers, empty)."""
        if not obj_text:
            return False
        clean = obj_text.strip().lower()
        clean = re.sub(r"^[^\w]+|[^\w]+$", "", clean)
        if not clean or clean in PRONOUNS_AND_FILLERS:
            return False
        words = clean.split()
        if all(w in PRONOUNS_AND_FILLERS for w in words):
            return False
        # Disallow objects that are purely single pronouns or articles + pronoun
        if len(words) <= 2 and words[-1] in PRONOUNS_AND_FILLERS:
            return False
        return True

    def _normalize_action(self, verb_lemma: str, obj_text: Optional[str] = None) -> str:
        """Normalize generic action verbs into standard requirement actions."""
        verb_lower = verb_lemma.lower()
        if verb_lower in ACTION_NORM_MAP:
            if verb_lower == "tell":
                if obj_text and any(w in obj_text.lower() for w in ("code", "example", "list", "detail")):
                    return "provide"
                return "explain"
            if verb_lower == "use" and obj_text and "formula" in obj_text.lower():
                return "provide"
            return ACTION_NORM_MAP[verb_lower]
        return verb_lower

    def _clean_object_span(self, doc, dobj_token) -> Optional[str]:
        """Extract a clean object noun phrase span stopping at clause boundaries."""
        subtree = list(dobj_token.subtree)
        span_tokens = []
        for t in subtree:
            # Stop if we hit a clause marker or conjunction after the direct object token
            if t.i > dobj_token.i and (
                t.dep_ in ("advcl", "relcl", "conj", "ccomp") or
                t.lower_ in (",", ";", "where", "what", "how", "when", "so", "because", "and", "or")
            ):
                break
            span_tokens.append(t)

        if not span_tokens:
            span_tokens = [dobj_token]

        start = span_tokens[0].i
        end = span_tokens[-1].i + 1
        raw_text = doc[start:end].text.strip()

        # Remove leading indirect pronouns (e.g. "me ", "us ")
        raw_text = re.sub(r"^(me|us|you|him|her|them)\s+", "", raw_text, flags=re.I).strip()
        # Remove trailing punctuation
        raw_text = re.sub(r"[.,;:?!]+$", "", raw_text).strip()

        return raw_text if raw_text else None

    def _determine_mandatory(self, sentence_text: str, action: str) -> bool:
        """Determine whether a requirement is mandatory or optional."""
        sent_lower = sentence_text.lower()
        if any(opt in sent_lower for opt in OPTIONAL_MARKERS):
            return False
        return True

    # ------------------------------------------------------------------
    # Core Extraction Strategies
    # ------------------------------------------------------------------

    def _extract_via_spacy(self, prompt: str, nlp, seen_keys: set) -> list:
        from app.semantic.types import Requirement
        from app.semantic.utils.patterns import REQUIREMENT_VERBS

        doc = nlp(prompt[:5000])
        results = []

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Inspect all candidate verb tokens in the sentence
            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue

                verb_lemma = token.lemma_.lower()
                if verb_lemma not in REQUIREMENT_VERBS:
                    continue

                # Check if this verb token has a direct object
                dobjs = [c for c in token.children if c.dep_ in ("dobj", "attr", "oprd", "nsubjpass")]
                obj_text: Optional[str] = None

                if dobjs:
                    obj_text = self._clean_object_span(doc, dobjs[0])

                # If direct object is missing or invalid pronoun, check prepositional objects
                if not self._is_valid_object(obj_text):
                    preps = [c for c in token.children if c.dep_ == "prep"]
                    for prep in preps:
                        pobjs = [c for c in prep.children if c.dep_ == "pobj"]
                        if pobjs:
                            p_text = self._clean_object_span(doc, pobjs[0])
                            if self._is_valid_object(p_text):
                                obj_text = f"{prep.text} {p_text}"
                                break

                # Check clausal complements if still no direct object
                if not self._is_valid_object(obj_text):
                    xcomps = [c for c in token.children if c.dep_ in ("xcomp", "ccomp")]
                    if xcomps:
                        clausal_text = doc[xcomps[0].left_edge.i : xcomps[0].right_edge.i + 1].text.strip()
                        clausal_text = clausal_text[:60].strip()
                        if self._is_valid_object(clausal_text):
                            obj_text = clausal_text

                # Validate object - if object is invalid pronoun/filler (e.g. "he", "it"), skip
                if not self._is_valid_object(obj_text):
                    continue

                action = self._normalize_action(verb_lemma, obj_text)
                dedup_key = (action.lower(), (obj_text or "").lower())
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                mandatory = self._determine_mandatory(sent_text, action)
                evidence = sent_text[:150]

                results.append(Requirement(
                    action=action,
                    object=obj_text,
                    mandatory=mandatory,
                    confidence=0.92,
                    method="spacy_dependency_rule",
                    evidence=evidence,
                ))

        return results

    def _extract_question_requirements(self, prompt: str, nlp, seen_keys: set) -> list:
        """Handle question-style requirements and embedded question clauses."""
        from app.semantic.types import Requirement

        results = []
        doc = nlp(prompt[:5000])

        for sent in doc.sents:
            sent_text = sent.text.strip()

            # 1. "How do I calculate accuracy?" -> action: calculate, object: accuracy
            m_how = re.search(r"\bhow\s+(can|do|should|to|are|is)\s+(?:i|we|you)?\s*(calculate|compute|find|solve|implement|explain|build)\s+([a-zA-Z0-9_\s\-\+]+)\b", sent_text, re.I)
            if m_how:
                action = m_how.group(2).lower()
                obj_text = m_how.group(3).strip().rstrip("?")
                if self._is_valid_object(obj_text):
                    dedup_key = (action, obj_text.lower())
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        results.append(Requirement(
                            action=action,
                            object=obj_text,
                            mandatory=True,
                            confidence=0.94,
                            method="question_rule",
                            evidence=m_how.group(0),
                        ))

            # 2. "how are we calculating" -> action: calculate, object: calculation methodology
            m_calc = re.search(r"\bhow\s+(?:are|is|do)\s+(?:we|i|you)\s+calculating\b", sent_text, re.I)
            if m_calc:
                dedup_key = ("calculate", "calculation methodology")
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    results.append(Requirement(
                        action="calculate",
                        object="calculation methodology",
                        mandatory=True,
                        confidence=0.90,
                        method="question_rule",
                        evidence=m_calc.group(0),
                    ))

            # 3. "What formula should I use?" -> action: provide, object: formula
            m_what_use = re.search(r"\bwhat\s+([a-zA-Z0-9_\s\-]+)\s+(should|can|do|to)\s+(?:i|we|you)\s+(use|apply|need)\b", sent_text, re.I)
            if m_what_use:
                obj_text = m_what_use.group(1).strip()
                action = "provide"
                if self._is_valid_object(obj_text):
                    dedup_key = (action, obj_text.lower())
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        results.append(Requirement(
                            action=action,
                            object=obj_text,
                            mandatory=True,
                            confidence=0.92,
                            method="question_rule",
                            evidence=m_what_use.group(0),
                        ))

            # 4. "What research can we do with this?" / "what research we can do" -> action: identify, object: research
            m_research = re.search(r"\bwhat\s+research\s+can\s+(?:we|i|you)\s+do\b", sent_text, re.I)
            if m_research:
                dedup_key = ("identify", "research possibilities")
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    results.append(Requirement(
                        action="identify",
                        object="research possibilities",
                        mandatory=True,
                        confidence=0.90,
                        method="question_rule",
                        evidence=m_research.group(0),
                    ))

            # 5. "where the formula comes from" -> action: explain, object: formula origin
            m_formula = re.search(r"\bwhere\s+the\s+formula\s+comes\s+from\b", sent_text, re.I)
            if m_formula:
                dedup_key = ("explain", "formula origin")
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    results.append(Requirement(
                        action="explain",
                        object="formula origin",
                        mandatory=True,
                        confidence=0.90,
                        method="question_rule",
                        evidence=m_formula.group(0),
                    ))

            # 6. "objective we can achieve" -> action: explain, object: achievable objectives
            m_obj = re.search(r"\bobjective\s+we\s+can\s+achieve\b", sent_text, re.I)
            if m_obj:
                dedup_key = ("explain", "achievable objectives")
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    results.append(Requirement(
                        action="explain",
                        object="achievable objectives",
                        mandatory=True,
                        confidence=0.90,
                        method="clause_rule",
                        evidence=m_obj.group(0),
                    ))

        return results

    def _extract_via_sentence_scan(self, prompt: str, seen_keys: set) -> list:
        from app.semantic.types import Requirement
        from app.semantic.utils.patterns import REQUIREMENT_VERBS

        results = []
        sentences = re.split(r"(?<=[.!?])\s+", prompt)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            m = re.match(
                r"^\s*(" + "|".join(re.escape(v) for v in REQUIREMENT_VERBS) + r")\b(.{0,80}?)(?:\.|$)",
                sent, re.I
            )
            if m:
                verb = m.group(1).lower()
                obj_raw = m.group(2).strip().rstrip(".")
                obj_clean = re.sub(r"^(a|an|the|me|us|you)\s+", "", obj_raw, flags=re.I).strip()
                if self._is_valid_object(obj_clean):
                    action = self._normalize_action(verb, obj_clean)
                    dedup_key = (action.lower(), obj_clean.lower())
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        results.append(Requirement(
                            action=action,
                            object=obj_clean,
                            mandatory=True,
                            confidence=0.75,
                            method="sentence_verb_rule",
                            evidence=sent[:120],
                        ))

        return results

    def _extract_via_patterns(self, prompt: str, seen_keys: set) -> list:
        from app.semantic.types import Requirement

        patterns = [
            (re.compile(r"\binclude\s+(time\s+)?complexity\b", re.I), "include", "time complexity", 0.97),
            (re.compile(r"\bprovide\s+a\s+dry\s+run\b", re.I), "provide", "dry run", 0.97),
            (re.compile(r"\bperform\s+a\s+dry\s+run\b", re.I), "perform", "dry run", 0.97),
            (re.compile(r"\binclude\s+a\s+dry\s+run\b", re.I), "include", "dry run", 0.97),
            (re.compile(r"\bgive\s+(a\s+)?python\s+(code\s+)?example\b", re.I), "provide", "Python code example", 0.97),
            (re.compile(r"\binclude\s+(a\s+)?python\s+(code|implementation)\b", re.I), "include", "Python code", 0.96),
            (re.compile(r"\bexplain\s+every\s+step\b", re.I), "explain", "every step", 0.96),
            (re.compile(r"\bexplain\s+the\s+code\b", re.I), "explain", "the code", 0.95),
            (re.compile(r"\bexplain\s+time\s+and\s+space\s+complexity\b", re.I), "explain", "time and space complexity", 0.97),
            (re.compile(r"\bwalk\s+(me\s+)?through\b", re.I), "explain", "walkthrough", 0.88),
            (re.compile(r"\bstep\s+by\s+step\b", re.I), "provide", "step-by-step explanation", 0.85),
        ]

        results = []
        for pattern, action, obj, confidence in patterns:
            m = pattern.search(prompt)
            if m:
                evidence = m.group(0).strip()
                dedup_key = (action.lower(), obj.lower() if obj else "")
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    results.append(Requirement(
                        action=action,
                        object=obj,
                        mandatory=True,
                        confidence=confidence,
                        method="regex_pattern",
                        evidence=evidence,
                    ))
        return results


def get_requirement_extractor() -> BaseRequirementExtractor:
    return SpacyRequirementExtractor()
