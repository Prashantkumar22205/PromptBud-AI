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

# Location of the V2 balanced development dataset, resolved relative to this file.
_backend_dir = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_PATH = _backend_dir / "data" / "intent_samples_v2_balanced.json"
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = _backend_dir / "data" / "intent_samples.json"


def load_training_data(data_path: Path | str | None = None) -> Tuple[List[str], List[str]]:
    """
    Load texts and labels from the specified or default dataset JSON file.

    Returns
    -------
    texts : list of str
    labels : list of str
    """
    target_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    if not target_path.exists():
        logger.warning("Intent dataset not found at %s; returning empty training data.", target_path)
        return [], []

    with target_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    samples = raw.get("samples", [])
    texts = [s["text"] for s in samples]
    labels = [s["intent"] if "intent" in s else s.get("gold_intent", "other") for s in samples]

    logger.debug("Loaded %d training samples from %s", len(samples), target_path)
    return texts, labels

