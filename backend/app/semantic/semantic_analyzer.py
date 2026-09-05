"""Main semantic analyzer module orchestrating all semantic sub-analyzers.

Phase 3A implementation: Rule/NLP layer + trained TF-IDF Logistic Regression intent classifier.
Module 3 ANALYZES the prompt ONLY. It NEVER rewrites, optimizes, or modifies the prompt.
"""

import logging
from typing import Optional
from app.semantic.types import SemanticAnalysis
from app.semantic.intent.classifier import get_intent_classifier, BaseIntentClassifier
from app.semantic.context.extractor import SpacyContextExtractor, BaseContextExtractor
from app.semantic.requirements.extractor import SpacyRequirementExtractor, BaseRequirementExtractor
from app.semantic.constraints.extractor import RuleConstraintExtractor, BaseConstraintExtractor
from app.semantic.ambiguity.detector import RuleAmbiguityDetector, BaseAmbiguityDetector

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Orchestrator for semantic analysis of prompts."""

    def __init__(
        self,
        intent_classifier: Optional[BaseIntentClassifier] = None,
        context_extractor: Optional[BaseContextExtractor] = None,
        requirement_extractor: Optional[BaseRequirementExtractor] = None,
        constraint_extractor: Optional[BaseConstraintExtractor] = None,
        ambiguity_detector: Optional[BaseAmbiguityDetector] = None,
    ):
        """Initialize SemanticAnalyzer with sub-modules or defaults."""
        self.intent_classifier = intent_classifier or get_intent_classifier()
        self.context_extractor = context_extractor or SpacyContextExtractor()
        self.requirement_extractor = requirement_extractor or SpacyRequirementExtractor()
        self.constraint_extractor = constraint_extractor or RuleConstraintExtractor()
        self.ambiguity_detector = ambiguity_detector or RuleAmbiguityDetector()

    def analyze(self, prompt: str) -> SemanticAnalysis:
        """Perform full semantic analysis on a prompt without modifying it.

        Args:
            prompt: Original prompt text.

        Returns:
            SemanticAnalysis containing intent, context, requirements, constraints, ambiguities.
        """
        # 1. Classify intent
        intent_result = self.intent_classifier.classify(prompt)

        # 2. Extract context (topic, domain, audience, language, prior knowledge)
        context_result = self.context_extractor.extract(prompt)

        # 3. Extract requirements (action items, desired outputs)
        requirements = self.requirement_extractor.extract(prompt)

        # 4. Extract constraints (formatting, length, tone, negative constraints)
        constraints = self.constraint_extractor.extract(prompt)

        # 5. Detect ambiguities, conflicts, and missing context
        ambiguities = self.ambiguity_detector.detect(
            prompt=prompt,
            constraints=constraints,
            context=context_result,
        )

        return SemanticAnalysis(
            intent=intent_result,
            context=context_result,
            requirements=requirements,
            constraints=constraints,
            ambiguities=ambiguities,
        )


def analyze_semantics(prompt: str) -> SemanticAnalysis:
    """Convenience function for semantic analysis.

    Args:
        prompt: Raw prompt text.

    Returns:
        SemanticAnalysis result.
    """
    analyzer = SemanticAnalyzer()
    return analyzer.analyze(prompt)
