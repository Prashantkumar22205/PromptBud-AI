"""
semantic/requirements/extractor.py

Requirement extraction from prompts.

A requirement is something the prompt explicitly asks for — something the
future optimizer MUST preserve.

Detection approach (Phase 3A)
-----------------------------
1. spaCy dependency parsing: find ROOT verbs and their dobj (direct objects).
2. Action-verb scanning: scan for known instruction verbs + their objects.
3. Explicit requirement patterns: regex for "Include X", "Provide Y", etc.

No LLM API is called.

Phase 3B extension
------------------
A BaseRequirementExtractor interface is defined here.
Future implementations can use semantic similarity or NLI to detect
implicit requirements not covered by explicit verbs.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum confidence for a verb-object pair to be reported
MIN_CONFIDENCE = 0.65


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
# Phase 3A implementation
# ---------------------------------------------------------------------------


class SpacyRequirementExtractor(BaseRequirementExtractor):
    """
    Extracts requirements using spaCy dependency parsing and regex patterns.
    """

    def extract(self, prompt: str) -> list:
        from app.semantic.types import Requirement
        from app.semantic.utils.patterns import REQUIREMENT_VERBS

        results: List[Requirement] = []
        seen_evidence = set()

        # Strategy 1: spaCy dependency parse
        nlp = _get_nlp()
        if nlp:
            results.extend(self._extract_via_spacy(prompt, nlp, seen_evidence))

        # Strategy 2: Sentence-level verb scan (fallback + supplementary)
        results.extend(self._extract_via_sentence_scan(prompt, seen_evidence))

        # Strategy 3: Explicit requirement patterns
        results.extend(self._extract_via_patterns(prompt, seen_evidence))

        return results

    # ------------------------------------------------------------------

    def _extract_via_spacy(self, prompt: str, nlp, seen: set) -> list:
        from app.semantic.types import Requirement
        from app.semantic.utils.patterns import REQUIREMENT_VERBS

        doc = nlp(prompt[:5000])
        results = []

        for sent in doc.sents:
            # Find the root verb of each sentence
            root = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
            if root is None:
                continue

            verb_lemma = root.lemma_.lower()
            if verb_lemma not in REQUIREMENT_VERBS:
                continue

            # Find direct object
            dobj = next((t for t in root.children if t.dep_ in ("dobj", "attr", "nsubj")), None)
            obj_text: Optional[str] = None
            if dobj:
                # Expand to include the full noun phrase
                obj_span = doc[dobj.left_edge.i : dobj.right_edge.i + 1]
                obj_text = obj_span.text.strip()

            evidence = sent.text.strip()
            if evidence in seen:
                continue
            seen.add(evidence)

            results.append(Requirement(
                action=verb_lemma,
                object=obj_text,
                mandatory=True,
                confidence=0.90,
                method="spacy_dependency_rule",
                evidence=evidence,
            ))

        return results

    def _extract_via_sentence_scan(self, prompt: str, seen: set) -> list:
        """Sentence-level action verb scan for when spaCy is unavailable or misses."""
        from app.semantic.types import Requirement
        from app.semantic.utils.patterns import REQUIREMENT_VERBS

        results = []
        sentences = re.split(r"(?<=[.!?])\s+", prompt)

        for sent in sentences:
            sent = sent.strip()
            if not sent or sent in seen:
                continue

            # Check if sentence starts with or contains a requirement verb
            m = re.match(
                r"^\s*(" + "|".join(re.escape(v) for v in REQUIREMENT_VERBS) + r")\b(.{0,80}?)(?:\.|$)",
                sent, re.I
            )
            if m:
                verb = m.group(1).lower()
                obj_raw = m.group(2).strip().rstrip(".")
                # Strip leading articles
                obj_clean = re.sub(r"^(a|an|the|me)\s+", "", obj_raw, flags=re.I).strip() or None
                if obj_clean and len(obj_clean) > 2:
                    seen.add(sent)
                    results.append(Requirement(
                        action=verb,
                        object=obj_clean if obj_clean else None,
                        mandatory=True,
                        confidence=0.75,
                        method="sentence_verb_rule",
                        evidence=sent[:120],
                    ))

        return results

    def _extract_via_patterns(self, prompt: str, seen: set) -> list:
        """Specific high-confidence requirement patterns."""
        from app.semantic.types import Requirement

        patterns = [
            (re.compile(r"\binclude\s+(time\s+)?complexity\b", re.I), "include", "time complexity", 0.97),
            (re.compile(r"\bprovide\s+a\s+dry\s+run\b", re.I), "provide", "dry run", 0.97),
            (re.compile(r"\bperform\s+a\s+dry\s+run\b", re.I), "perform", "dry run", 0.97),
            (re.compile(r"\binclude\s+a\s+dry\s+run\b", re.I), "include", "dry run", 0.97),
            (re.compile(r"\bgive\s+(a\s+)?python\s+(code\s+)?example\b", re.I), "give", "Python code example", 0.97),
            (re.compile(r"\binclude\s+(a\s+)?python\s+(code|implementation)\b", re.I), "include", "Python code", 0.96),
            (re.compile(r"\bexplain\s+every\s+step\b", re.I), "explain", "every step", 0.96),
            (re.compile(r"\bexplain\s+the\s+code\b", re.I), "explain", "the code", 0.95),
            (re.compile(r"\bexplain\s+time\s+and\s+space\s+complexity\b", re.I), "explain", "time and space complexity", 0.97),
            (re.compile(r"\bwalk\s+(me\s+)?through\b", re.I), "walk through", None, 0.88),
            (re.compile(r"\bstep\s+by\s+step\b", re.I), "provide", "step-by-step explanation", 0.85),
        ]

        results = []
        for pattern, action, obj, confidence in patterns:
            m = pattern.search(prompt)
            if m:
                evidence = m.group(0).strip()
                if evidence not in seen:
                    seen.add(evidence)
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
