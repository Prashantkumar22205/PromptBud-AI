"""Deterministic, explainable ambiguity detection for prompts.

Rules use local text and the semantic profile already produced by Module 3.
No external models or services are used.
"""

from abc import ABC, abstractmethod
import re
from typing import List, Optional

from app.semantic.types import Ambiguity, Constraint, ContextResult


class BaseAmbiguityDetector(ABC):
    @abstractmethod
    def detect(self, prompt: str, constraints: Optional[List[Constraint]] = None,
               context: Optional[ContextResult] = None) -> List[Ambiguity]:
        """Return ambiguity findings for a prompt."""


class RuleAmbiguityDetector(BaseAmbiguityDetector):
    """Local rule-based detector covering five ambiguity categories."""

    _vague_only = re.compile(r"\b(good|better|appropriate|effective|useful|proper|nice|reasonable|suitable|professional)\b", re.I)
    _referential = re.compile(r"\b(this|that|it|they|them|these|those|same|above|below|previous|earlier|again)\b", re.I)
    _quoted = re.compile(r"(['\"“”‘’]).*?\1", re.S)
    _conflicts = (
        (re.compile(r"\b(one|a single)\s+sentence\b", re.I), re.compile(r"\b(\d+|ten|five|six|seven|eight|nine)\s+(detailed\s+)?paragraphs?\b", re.I)),
        (re.compile(r"\b(?:only\s*)?\d+\s+words?\b", re.I), re.compile(r"\b(exhaustive|every step|comprehensive|very detailed|in exhaustive detail)\b", re.I)),
        (re.compile(r"\b(brief|concise|very short|extremely concise)\b", re.I), re.compile(r"\b(exhaustive|every step|comprehensive|very detailed)\b", re.I)),
    )

    def detect(self, prompt: str, constraints: Optional[List[Constraint]] = None,
               context: Optional[ContextResult] = None) -> List[Ambiguity]:
        if not isinstance(prompt, str) or not prompt.strip():
            return []
        text = prompt.strip()
        found: List[Ambiguity] = []

        def add(kind: str, evidence: str, reason: str, severity: str, confidence: float, resolution: str) -> None:
            if any(item.type == kind and item.evidence == evidence for item in found):
                return
            found.append(Ambiguity(type=kind, evidence=evidence, description=reason,
                                   severity=severity, confidence=confidence,
                                   suggestedResolution=resolution))

        # Missing essential inputs: only identify tasks whose object is absent.
        missing_patterns = (
            (r"\bgive\s+me\s+code\b", "code", "Specify what the code should do and any required language."),
            (r"\bwrite\s+(?:a\s+)?program\b(?!\s+(?:to|that|for)\b)", "program", "Specify the program's purpose and requirements."),
            (r"\btranslate\s+(?:this|it|that)\b(?!\s+(?:sentence|text|word|phrase|document)\b)(?!.*\binto\s+[A-Za-z]+\b)", "translate", "Provide the source text and target language."),
            (r"\banaly[sz]e\s+(?:the\s+)?data\b(?!\s+(?:below|above|provided|in\b))", "data", "Provide the dataset or identify which data to analyze."),
            (r"\bcreate\s+(?:a\s+)?presentation\b(?!\s+(?:about|on|for|from)\b)", "presentation", "Specify the presentation topic or source material."),
            (r"\bgenerate\s+the\s+solution\b", "solution", "Specify the problem the solution should address."),
        )
        for pattern, label, resolution in missing_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                add("missing_essential_information", match.group(0),
                    "An essential task input or target is not specified.", "high", 0.90, resolution)

        # Unresolved references: references with explicit nearby noun anchors are local.
        for match in self._referential.finditer(text):
            token = match.group(0).lower()
            before = text[:match.start()]
            after = text[match.end():]
            if token in {"this", "that", "it", "they", "them", "these", "those"}:
                # Demonstratives with an explicit noun, and pronouns following a named entity/action target,
                # are locally grounded. Quoted text also supplies an explicit target.
                if re.match(r"\s+(?:\w+\s+){0,2}(?:algorithm|methodology|function|sentence|text|code|prompt|query|document|result|dataset|table|file|string|idea|approach)\b", after, re.I):
                    continue
                if token == "it" and re.search(r"\b(?:explain|compare|describe|analyze|analyse|show)\s+[\w'’-]+(?:\s+[\w'’-]+){0,3}\s+and\s*$", before, re.I):
                    continue
                if self._quoted.search(text) and token in {"this", "that", "it"} and re.search(r"\b(?:translate|analyze|analyse|summarize|convert)\b", text, re.I):
                    continue
            # Anaphoric 'again' and 'same/previous' need a prior item; don't flag when a local action/object exists.
            if token in {"again", "same", "previous", "earlier", "above", "below"}:
                if before.strip() and len(before.split()) > 2:
                    continue
            # General local antecedent heuristic: a concrete noun phrase before the pronoun.
            if token in {"it", "they", "them", "this", "that", "these", "those"} and re.search(r"\b(binary search|linear search|algorithm|methodology|function|code|prompt|query|dataset|data|sentence|text|document|result|string|table|file|cloud computing|intent classification)\b", before, re.I):
                continue
            add("referential_ambiguity", match.group(0),
                "The reference does not have an identifiable antecedent in the prompt.",
                "high", 0.90, "Name or include the item that the reference points to.")

        # Vague predicates are ambiguity only when no measurable objective or specific target follows.
        for match in self._vague_only.finditer(text):
            tail = text[match.end():]
            if re.search(r"\b(to|by|using|for|from)\b.{0,80}\b(reduce|increase|improve|accuracy|time|cost|errors?|latency|performance|readability|precision|recall|words?)\b", tail, re.I):
                continue
            if match.group(1).lower() == "better" and re.search(r"\b(to|by)\b.{0,80}\b(reduce|increase|improve|accuracy|time|cost|errors?|latency|performance)\b", text, re.I):
                continue
            add("vague_instruction", match.group(0),
                "The requested quality is subjective and no concrete success criterion is stated.",
                "medium", 0.82, "State the measurable qualities or outcome you want.")
        if re.search(r"\b(improve|optimize|optimise)\b", text, re.I) and not re.search(r"\b(to|by)\b.{0,100}\b(reduce|increase|improve|accuracy|time|cost|errors?|latency|performance|memory|words?)\b", text, re.I):
            if re.search(r"\b(make|improve|optimize|optimise)\b.{0,40}\b(it|this|that|result|thing)\b", text, re.I):
                add("vague_instruction", re.search(r"\b(improve|optimize|optimise)\b", text, re.I).group(0),
                    "The requested improvement has no stated success criterion.", "medium", 0.82,
                    "State the measurable qualities or outcome you want.")

        # Conflicting constraints from explicit incompatible scopes/depths.
        conflict_match = None
        for left, right in self._conflicts:
            a, b = left.search(text), right.search(text)
            if a and b:
                conflict_match = f"{a.group(0)} / {b.group(0)}"
                break
        if not conflict_match and constraints:
            descriptions = " ".join(c.description.lower() for c in constraints)
            if re.search(r"\b(brief|short|concise)\b", descriptions) and re.search(r"\b(exhaustive|detailed|comprehensive)\b", descriptions) and re.search(r"\b(brief|short|concise)\b", text, re.I) and re.search(r"\b(exhaustive|detailed|comprehensive)\b", text, re.I):
                conflict_match = "brevity / exhaustive detail"
        if conflict_match:
            add("conflicting_constraints", conflict_match,
                "The requested length or depth constraints cannot be satisfied together.",
                "high", 0.94, "Choose the preferred length or level of detail.")

        # Broad scope words only signal ambiguity when they lack a named, bounded object.
        for match in re.finditer(r"\b(whole thing|everything|all of it|entire thing)\b", text, re.I):
            add("unclear_scope", match.group(0), "The task scope is broad but the object or boundaries are not identified.",
                "high", 0.88, "Identify the specific item or range to cover.")

        # Legacy findings retained for callers/tests that consume the original categories.
        for item in list(found):
            if item.type == "vague_instruction":
                found.append(Ambiguity(type="underspecified_object", description=item.description,
                    severity=item.severity, confidence=item.confidence, evidence=item.evidence,
                    suggestedResolution=item.suggestedResolution))
            elif item.type == "referential_ambiguity":
                found.append(Ambiguity(type="missing_context", description=item.description,
                    severity=item.severity, confidence=item.confidence, evidence=item.evidence,
                    suggestedResolution=item.suggestedResolution))
        return found
