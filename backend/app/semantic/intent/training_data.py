"""
semantic/intent/training_data.py

Loads the intent classification training data from the seed JSON file.

IMPORTANT
---------
The file data/intent_samples.json is a DEVELOPMENT SEED DATASET.
It is NOT the final research evaluation dataset.
It exists to bootstrap the classifier during development.

To replace it with a larger curated dataset:
  1. Update data/intent_samples.json (or point DATA_PATH to a new file).
  2. Retrain by calling IntentClassifier().fit().
  3. No code changes needed in this file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Location of the seed dataset, resolved relative to this file.
_backend_dir = Path(__file__).parent.parent.parent.parent
DATA_PATH = _backend_dir / "data" / "intent_samples.json"
if not DATA_PATH.exists():
    # Try project root / data / intent_samples.json as fallback
    DATA_PATH = _backend_dir.parent / "data" / "intent_samples.json"


def load_training_data() -> Tuple[List[str], List[str]]:
    """
    Load texts and labels from the seed JSON file.

    Returns
    -------
    texts : list of str
    labels : list of str
    """
    if not DATA_PATH.exists():
        logger.warning("Intent seed dataset not found at %s; returning empty training data.", DATA_PATH)
        return [], []

    with DATA_PATH.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    samples = raw.get("samples", [])
    texts = [s["text"] for s in samples]
    labels = [s["intent"] for s in samples]

    logger.debug("Loaded %d training samples from %s", len(samples), DATA_PATH)
    return texts, labels
