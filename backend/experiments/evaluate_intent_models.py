"""
experiments/evaluate_intent_models.py

Reproducible Research Evaluation Script comparing:
  E1: V2 Dev Dataset → TF-IDF → Logistic Regression
  E2: V2 Dev Dataset → SentenceTransformer(all-MiniLM-L6-v2) → Logistic Regression

Evaluation Protocol
-------------------
1. Development Validation: 5-Fold Stratified Cross-Validation on V2 Development Dataset (160 samples).
2. Final Model Training: Fit E1 and E2 on all 160 V2 development samples.
3. Frozen Gold Evaluation: Evaluate final E1 and E2 models on Frozen Gold Set v1.1 (240 samples).
4. Output Generation: Save CSV/JSON metrics, confusion matrices, and comparison summary to experiments/results/.

Strict Rules Applied:
  - Gold set is NEVER trained on, modified, or used for threshold tuning.
  - Random seeds fixed at random_state=42 for exact reproducibility.
  - Development validation metrics and Gold test metrics are reported separately.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

# Ensure backend root is in sys.path
import sys
_backend_root = Path(__file__).parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.semantic.intent.classifier import (
    TfidfLogisticIntentClassifier,
    SemanticIntentClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = _backend_root / "data"
RESULTS_DIR = _backend_root / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEV_PATH = DATA_DIR / "intent_samples_v2_balanced.json"
GOLD_PATH = DATA_DIR / "gold_intent_samples_v1_1.json"

TAXONOMY = [
    "educational_explanation",
    "question_answering",
    "code_generation",
    "code_debugging",
    "code_explanation",
    "summarization",
    "translation",
    "rewriting",
    "comparison",
    "data_analysis",
    "creative_writing",
    "classification",
    "information_extraction",
    "problem_solving",
    "content_generation",
    "other",
]


def load_dataset(path: Path) -> Tuple[List[Dict], List[str], List[str]]:
    """Load JSON dataset file."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    samples = data.get("samples", [])
    texts = [s["text"] for s in samples]
    labels = [s.get("intent") or s.get("gold_intent") for s in samples]
    return samples, texts, labels


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    """Compute overall accuracy, macro precision, macro recall, macro F1."""
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f1), 4),
    }


def run_cross_validation(texts: List[str], labels: List[str]) -> Tuple[Dict, Dict]:
    """Run 5-Fold Stratified CV for E1 and E2 on Development Dataset."""
    logger.info("--- Starting 5-Fold Stratified Cross-Validation on Development Set (160 samples) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    e1_preds_all, e2_preds_all, y_true_all = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # E1: TF-IDF + LR
        clf_e1 = TfidfLogisticIntentClassifier()
        clf_e1.fit(train_texts, train_labels)
        val_e1_preds = [clf_e1.predict(t)[0] for t in val_texts]

        # E2: SentenceTransformer + LR
        clf_e2 = SemanticIntentClassifier()
        clf_e2.fit(train_texts, train_labels)
        val_e2_preds = [clf_e2.predict(t)[0] for t in val_texts]

        e1_preds_all.extend(val_e1_preds)
        e2_preds_all.extend(val_e2_preds)
        y_true_all.extend(val_labels)

    e1_cv_metrics = compute_metrics(y_true_all, e1_preds_all, TAXONOMY)
    e2_cv_metrics = compute_metrics(y_true_all, e2_preds_all, TAXONOMY)

    logger.info("E1 CV Results: %s", e1_cv_metrics)
    logger.info("E2 CV Results: %s", e2_cv_metrics)
    return e1_cv_metrics, e2_cv_metrics


def evaluate_gold(
    clf_e1: TfidfLogisticIntentClassifier,
    clf_e2: SemanticIntentClassifier,
    gold_samples: List[Dict],
) -> Tuple[Dict, Dict, List[Dict], Dict, Dict]:
    """Evaluate final trained E1 and E2 models on Frozen Gold Evaluation Set v1.1."""
    logger.info("--- Evaluating Final E1 and E2 Models on Frozen Gold Set v1.1 (240 samples) ---")
    
    gold_texts = [s["text"] for s in gold_samples]
    gold_labels = [s["gold_intent"] for s in gold_samples]
    gold_subsets = [s["subset"] for s in gold_samples]

    # Measure E1 Inference Time
    t0 = time.perf_counter()
    e1_preds, e1_confs, e1_scores_all = [], [], []
    for t in gold_texts:
        label, conf, scores = clf_e1.predict(t)
        e1_preds.append(label)
        e1_confs.append(conf)
        e1_scores_all.append(scores)
    e1_latency = ((time.perf_counter() - t0) / len(gold_texts)) * 1000.0

    # Measure E2 Inference Time
    t0 = time.perf_counter()
    e2_preds, e2_confs, e2_scores_all = [], [], []
    for t in gold_texts:
        label, conf, scores = clf_e2.predict(t)
        e2_preds.append(label)
        e2_confs.append(conf)
        e2_scores_all.append(scores)
    e2_latency = ((time.perf_counter() - t0) / len(gold_texts)) * 1000.0

    e1_overall = compute_metrics(gold_labels, e1_preds, TAXONOMY)
    e1_overall["mean_latency_ms"] = round(e1_latency, 2)

    e2_overall = compute_metrics(gold_labels, e2_preds, TAXONOMY)
    e2_overall["mean_latency_ms"] = round(e2_latency, 2)

    # Subset analysis
    unique_subsets = sorted(list(set(gold_subsets)))
    for sub in unique_subsets:
        sub_indices = [i for i, s in enumerate(gold_subsets) if s == sub]
        sub_y = [gold_labels[i] for i in sub_indices]
        sub_e1_p = [e1_preds[i] for i in sub_indices]
        sub_e2_p = [e2_preds[i] for i in sub_indices]

        e1_overall[f"subset_{sub}_acc"] = round(float(accuracy_score(sub_y, sub_e1_p)), 4)
        e2_overall[f"subset_{sub}_acc"] = round(float(accuracy_score(sub_y, sub_e2_p)), 4)

    # Predictions detail
    pred_records = []
    for idx, sample in enumerate(gold_samples):
        pred_records.append({
            "id": sample["id"],
            "text": sample["text"],
            "gold_intent": sample["gold_intent"],
            "subset": sample["subset"],
            "e1_pred": e1_preds[idx],
            "e1_conf": e1_confs[idx],
            "e2_pred": e2_preds[idx],
            "e2_conf": e2_confs[idx],
            "e1_correct": e1_preds[idx] == sample["gold_intent"],
            "e2_correct": e2_preds[idx] == sample["gold_intent"],
        })

    # Per-intent metrics
    p1, r1, f1_1, _ = precision_recall_fscore_support(gold_labels, e1_preds, labels=TAXONOMY, zero_division=0)
    p2, r2, f1_2, _ = precision_recall_fscore_support(gold_labels, e2_preds, labels=TAXONOMY, zero_division=0)

    per_intent = []
    for i, intent in enumerate(TAXONOMY):
        per_intent.append({
            "intent": intent,
            "e1_precision": round(float(p1[i]), 4),
            "e1_recall": round(float(r1[i]), 4),
            "e1_f1": round(float(f1_1[i]), 4),
            "e2_precision": round(float(p2[i]), 4),
            "e2_recall": round(float(r2[i]), 4),
            "e2_f1": round(float(f1_2[i]), 4),
            "f1_diff": round(float(f1_2[i] - f1_1[i]), 4),
        })

    # Confusion matrices
    cm1 = confusion_matrix(gold_labels, e1_preds, labels=TAXONOMY)
    cm2 = confusion_matrix(gold_labels, e2_preds, labels=TAXONOMY)

    return e1_overall, e2_overall, pred_records, per_intent, {"e1": cm1.tolist(), "e2": cm2.tolist()}


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    """Write list of dicts to CSV."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    logger.info("=== PromptOptAI Module 3: E1 vs E2 Research Evaluation ===")

    # 1. Load Development Dataset
    dev_samples, dev_texts, dev_labels = load_dataset(DEV_PATH)
    logger.info("Loaded %d V2 development samples from %s", len(dev_samples), DEV_PATH)

    # 2. Run Stratified Cross-Validation on Dev Data
    e1_cv, e2_cv = run_cross_validation(dev_texts, dev_labels)

    # 3. Train Final E1 & E2 Models on All V2 Dev Data
    logger.info("Training final E1 (TF-IDF + LR) on all 160 V2 dev samples...")
    clf_e1 = TfidfLogisticIntentClassifier()
    clf_e1.fit(dev_texts, dev_labels)

    logger.info("Training final E2 (SentenceTransformer + LR) on all 160 V2 dev samples...")
    clf_e2 = SemanticIntentClassifier()
    clf_e2.fit(dev_texts, dev_labels)

    # 4. Load Frozen Gold Set v1.1
    gold_samples, gold_texts, gold_labels = load_dataset(GOLD_PATH)
    logger.info("Loaded %d Frozen Gold v1.1 samples from %s", len(gold_samples), GOLD_PATH)

    # 5. Evaluate on Gold Set
    e1_gold, e2_gold, predictions, per_intent, confusion = evaluate_gold(clf_e1, clf_e2, gold_samples)

    # 6. Save Development CV Metrics
    cv_rows = [
        {"model": "E1 (TF-IDF+LR)", **e1_cv},
        {"model": "E2 (Semantic+LR)", **e2_cv},
        {
            "model": "Difference (E2 - E1)",
            "accuracy": round(e2_cv["accuracy"] - e1_cv["accuracy"], 4),
            "macro_precision": round(e2_cv["macro_precision"] - e1_cv["macro_precision"], 4),
            "macro_recall": round(e2_cv["macro_recall"] - e1_cv["macro_recall"], 4),
            "macro_f1": round(e2_cv["macro_f1"] - e1_cv["macro_f1"], 4),
        },
    ]
    save_csv(RESULTS_DIR / "development_metrics.csv", cv_rows, list(cv_rows[0].keys()))

    # 7. Save Gold Metrics
    gold_rows = [
        {"model": "E1 (TF-IDF+LR)", **e1_gold},
        {"model": "E2 (Semantic+LR)", **e2_gold},
        {
            "model": "Difference (E2 - E1)",
            "accuracy": round(e2_gold["accuracy"] - e1_gold["accuracy"], 4),
            "macro_precision": round(e2_gold["macro_precision"] - e1_gold["macro_precision"], 4),
            "macro_recall": round(e2_gold["macro_recall"] - e1_gold["macro_recall"], 4),
            "macro_f1": round(e2_gold["macro_f1"] - e1_gold["macro_f1"], 4),
            "mean_latency_ms": round(e2_gold["mean_latency_ms"] - e1_gold["mean_latency_ms"], 2),
            "subset_clear_in_scope_acc": round(e2_gold["subset_clear_in_scope_acc"] - e1_gold["subset_clear_in_scope_acc"], 4),
            "subset_boundary_acc": round(e2_gold["subset_boundary_acc"] - e1_gold["subset_boundary_acc"], 4),
            "subset_ambiguous_acc": round(e2_gold["subset_ambiguous_acc"] - e1_gold["subset_ambiguous_acc"], 4),
            "subset_OOS_acc": round(e2_gold["subset_OOS_acc"] - e1_gold["subset_OOS_acc"], 4),
            "subset_robustness_acc": round(e2_gold["subset_robustness_acc"] - e1_gold["subset_robustness_acc"], 4),
        },
    ]
    save_csv(RESULTS_DIR / "gold_metrics.csv", gold_rows, list(gold_rows[0].keys()))

    # 8. Save Per-Intent Metrics
    save_csv(RESULTS_DIR / "per_intent_metrics.csv", per_intent, list(per_intent[0].keys()))

    # 9. Save Predictions
    save_csv(RESULTS_DIR / "prediction_outputs.csv", predictions, list(predictions[0].keys()))

    # 10. Save Confusion Matrices
    with (RESULTS_DIR / "e1_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual/predicted"] + TAXONOMY)
        for idx, row in enumerate(confusion["e1"]):
            writer.writerow([TAXONOMY[idx]] + row)

    with (RESULTS_DIR / "e2_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual/predicted"] + TAXONOMY)
        for idx, row in enumerate(confusion["e2"]):
            writer.writerow([TAXONOMY[idx]] + row)

    # 11. Save Summary JSON
    summary_json = {
        "experiment_name": "Module 3 Intent Classification: E1 (TF-IDF) vs E2 (SentenceTransformer)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dev_dataset": "data/intent_samples_v2_balanced.json (160 samples)",
        "gold_dataset": "data/gold_intent_samples_v1_1.json (240 samples)",
        "models": {
            "E1": {
                "name": "TfidfLogisticIntentClassifier",
                "representation": "TF-IDF (ngram_range=(1,2), max_features=10000)",
                "classifier": "LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)",
            },
            "E2": {
                "name": "SemanticIntentClassifier",
                "representation": "SentenceTransformer('all-MiniLM-L6-v2', dim=384)",
                "classifier": "LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)",
            },
        },
        "development_cv_metrics": {
            "E1": e1_cv,
            "E2": e2_cv,
        },
        "gold_test_metrics": {
            "E1": e1_gold,
            "E2": e2_gold,
        },
    }

    with (RESULTS_DIR / "experiment_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    # 12. Save Comparison Summary CSV
    comparison_summary_rows = [
        {"Metric": "Overall Accuracy", "E1_TFIDF": f"{e1_gold['accuracy']*100:.2f}%", "E2_Semantic": f"{e2_gold['accuracy']*100:.2f}%", "Difference": f"{(e2_gold['accuracy']-e1_gold['accuracy'])*100:+.2f}%"},
        {"Metric": "Macro Precision", "E1_TFIDF": f"{e1_gold['macro_precision']*100:.2f}%", "E2_Semantic": f"{e2_gold['macro_precision']*100:.2f}%", "Difference": f"{(e2_gold['macro_precision']-e1_gold['macro_precision'])*100:+.2f}%"},
        {"Metric": "Macro Recall", "E1_TFIDF": f"{e1_gold['macro_recall']*100:.2f}%", "E2_Semantic": f"{e2_gold['macro_recall']*100:.2f}%", "Difference": f"{(e2_gold['macro_recall']-e1_gold['macro_recall'])*100:+.2f}%"},
        {"Metric": "Macro F1 Score", "E1_TFIDF": f"{e1_gold['macro_f1']*100:.2f}%", "E2_Semantic": f"{e2_gold['macro_f1']*100:.2f}%", "Difference": f"{(e2_gold['macro_f1']-e1_gold['macro_f1'])*100:+.2f}%"},
        {"Metric": "Clear In-Scope Accuracy", "E1_TFIDF": f"{e1_gold['subset_clear_in_scope_acc']*100:.2f}%", "E2_Semantic": f"{e2_gold['subset_clear_in_scope_acc']*100:.2f}%", "Difference": f"{(e2_gold['subset_clear_in_scope_acc']-e1_gold['subset_clear_in_scope_acc'])*100:+.2f}%"},
        {"Metric": "Boundary Subset Accuracy", "E1_TFIDF": f"{e1_gold['subset_boundary_acc']*100:.2f}%", "E2_Semantic": f"{e2_gold['subset_boundary_acc']*100:.2f}%", "Difference": f"{(e2_gold['subset_boundary_acc']-e1_gold['subset_boundary_acc'])*100:+.2f}%"},
        {"Metric": "Ambiguous Subset Accuracy", "E1_TFIDF": f"{e1_gold['subset_ambiguous_acc']*100:.2f}%", "E2_Semantic": f"{e2_gold['subset_ambiguous_acc']*100:.2f}%", "Difference": f"{(e2_gold['subset_ambiguous_acc']-e1_gold['subset_ambiguous_acc'])*100:+.2f}%"},
        {"Metric": "OOS Rejection Accuracy", "E1_TFIDF": f"{e1_gold['subset_OOS_acc']*100:.2f}%", "E2_Semantic": f"{e2_gold['subset_OOS_acc']*100:.2f}%", "Difference": f"{(e2_gold['subset_OOS_acc']-e1_gold['subset_OOS_acc'])*100:+.2f}%"},
        {"Metric": "Robustness Subset Accuracy", "E1_TFIDF": f"{e1_gold['subset_robustness_acc']*100:.2f}%", "E2_Semantic": f"{e2_gold['subset_robustness_acc']*100:.2f}%", "Difference": f"{(e2_gold['subset_robustness_acc']-e1_gold['subset_robustness_acc'])*100:+.2f}%"},
        {"Metric": "Mean Inference Latency (ms)", "E1_TFIDF": f"{e1_gold['mean_latency_ms']:.2f} ms", "E2_Semantic": f"{e2_gold['mean_latency_ms']:.2f} ms", "Difference": f"{(e2_gold['mean_latency_ms']-e1_gold['mean_latency_ms']):+.2f} ms"},
    ]
    save_csv(RESULTS_DIR / "comparison_summary.csv", comparison_summary_rows, ["Metric", "E1_TFIDF", "E2_Semantic", "Difference"])

    logger.info("=== Evaluation Complete. All results saved to %s ===", RESULTS_DIR)


if __name__ == "__main__":
    main()
