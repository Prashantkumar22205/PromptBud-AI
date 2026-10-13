"""
experiments/manual_test_context_topic.py

Manual test script verifying extracted topic, confidence, method, evidence,
and full context objects across the 5 required test prompts.
"""

import json
import sys
from pathlib import Path

_backend_root = Path(__file__).parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.semantic.context.extractor import SpacyContextExtractor

prompts = [
    "Explain binary search in C++ for a beginner.",
    "I am preparing for my DSA interview and I need you to explain how binary search works.",
    "Our guide wants us to explain the methodology for prompt optimization.",
    "Compare TF-IDF and Sentence Transformer embeddings for intent classification.",
    "Do it again.",
]

extractor = SpacyContextExtractor()

for idx, p in enumerate(prompts, 1):
    print(f"\n==================================================")
    print(f"TEST PROMPT {idx}: \"{p}\"")
    print(f"==================================================")
    res = extractor.extract(p)
    if res.topic:
        print(f"Extracted Topic Value: \"{res.topic.value}\"")
        print(f"Confidence:            {res.topic.confidence}")
        print(f"Method:                \"{res.topic.method}\"")
        print(f"Evidence:              \"{res.topic.evidence}\"")
    else:
        print("Extracted Topic Value: null (None)")
    print("\nComplete Context Object:")
    print(json.dumps(res.model_dump(), indent=2))
