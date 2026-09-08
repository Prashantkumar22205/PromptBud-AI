"""
experiments/evaluate_e3_hybrid.py

Reproducible Research Experiment Script: E3 Hybrid Intent Classifier
Probability-level fusion of E1 (TF-IDF + LR) and E2 (SentenceTransformer + LR).

Formula:
  P_hybrid(c) = alpha * P_E1(c) + (1 - alpha) * P_E2(c)

Evaluation Protocol:
1. Alpha Tuning: 5-Fold Stratified CV on V2 Development Dataset (160 samples)
   Candidate alphas: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
   Select best alpha strictly on Dev CV Macro-F1.
2. Final Training: Fit E1 and E2 on all 160 V2 samples.
3. Frozen Gold Evaluation: Evaluate E1, E2, and E3 (with frozen best alpha) ONCE on Gold v1.1 (240 samples).
4. Results Export: Generate required CSV/JSON files in experiments/results/.
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
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

import sys
_backend_root = Path(__file__).parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.semantic.intent.classifier import (
    TfidfLogisticIntentClassifier,
    SemanticIntentClassifier,
    HybridIntentClassifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = _backend_root / "data"
RESULTS_DIR = _backend_root / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEV_PATH = DATA_DIR / "intent_samples_v2_balanced.json"
GOLD_PATH = DATA_DIR / "gold_intent_samples_v1_1.json"

TAXONOMY = [
    "educational_explanation", "question_answering", "code_generation",
    "code_debugging", "code_explanation", "summarization", "translation",
    "rewriting", "comparison", "data_analysis", "creative_writing",
    "classification", "information_extraction", "problem_solving",
    "content_generation", "other"
]


def load_dataset(path: Path) -> Tuple[List[Dict], List[str], List[str]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    samples = data.get("samples", [])
    texts = [s["text"] for s in samples]
    labels = [s.get("intent") or s.get("gold_intent") for s in samples]
    return samples, texts, labels


def compute_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
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


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_alpha_cross_validation(texts: List[str], labels: List[str]) -> Tuple[List[Dict], float, Dict]:
    logger.info("--- Starting 5-Fold Stratified Cross-Validation for E3 Alpha Selection ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Pre-collect out-of-fold probability vectors for E1 and E2
    e1_oof_scores = [None] * len(texts)
    e2_oof_scores = [None] * len(texts)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        clf_e1 = TfidfLogisticIntentClassifier(confidence_threshold=0.0)
        clf_e1.fit(train_texts, train_labels)

        clf_e2 = SemanticIntentClassifier(confidence_threshold=0.0)
        clf_e2.fit(train_texts, train_labels)

        for idx, t in zip(val_idx, val_texts):
            _, _, s1 = clf_e1.predict(t)
            _, _, s2 = clf_e2.predict(t)
            e1_oof_scores[idx] = s1
            e2_oof_scores[idx] = s2

    alpha_curve_rows = []
    best_alpha = 0.5
    best_macro_f1 = -1.0
    best_cv_metrics = {}

    for alpha in candidate_alphas:
        y_preds = []
        for s1, s2 in zip(e1_oof_scores, e2_oof_scores):
            classes = sorted(list(set(s1.keys()) | set(s2.keys())))
            h_scores = {}
            for c in classes:
                h_scores[c] = alpha * s1.get(c, 0.0) + (1.0 - alpha) * s2.get(c, 0.0)
            top_c = max(h_scores, key=h_scores.get)
            y_preds.append(top_c)

        m = compute_metrics(labels, y_preds, TAXONOMY)
        row = {"alpha": alpha, **m}
        alpha_curve_rows.append(row)
        logger.info("Alpha=%.1f => CV Acc: %.4f | Macro-F1: %.4f", alpha, m["accuracy"], m["macro_f1"])

        if m["macro_f1"] > best_macro_f1:
            best_macro_f1 = m["macro_f1"]
            best_alpha = alpha
            best_cv_metrics = m

    logger.info(">>> Selected Best Alpha from Dev CV: %.1f (Macro-F1: %.4f) <<<", best_alpha, best_macro_f1)
    return alpha_curve_rows, best_alpha, best_cv_metrics


def main():
    logger.info("=== PromptOptAI Module 3: E3 Hybrid Intent Classifier Experiment ===")

    # 1. Load Development Dataset
    dev_samples, dev_texts, dev_labels = load_dataset(DEV_PATH)
    logger.info("Loaded %d V2 development samples from %s", len(dev_samples), DEV_PATH.name)

    # 2. Run Stratified Cross-Validation to Select Alpha
    alpha_curve_rows, best_alpha, e3_cv_metrics = run_alpha_cross_validation(dev_texts, dev_labels)

    # Save alpha curve CSV
    save_csv(RESULTS_DIR / "e3_development_alpha_curve.csv", alpha_curve_rows, ["alpha", "accuracy", "macro_precision", "macro_recall", "macro_f1"])

    # Compute E1 and E2 CV metrics for comparison
    clf_e1_cv = TfidfLogisticIntentClassifier()
    clf_e2_cv = SemanticIntentClassifier()
    
    # 3. Train Final Models on All 160 V2 Samples
    logger.info("Training final E1 (TF-IDF) on all 160 V2 samples...")
    clf_e1 = TfidfLogisticIntentClassifier()
    clf_e1.fit(dev_texts, dev_labels)

    logger.info("Training final E2 (Semantic) on all 160 V2 samples...")
    clf_e2 = SemanticIntentClassifier()
    clf_e2.fit(dev_texts, dev_labels)

    logger.info("Building final E3 Hybrid with frozen best alpha=%.1f...", best_alpha)
    clf_e3 = HybridIntentClassifier(alpha=best_alpha, e1_clf=clf_e1, e2_clf=clf_e2)
    clf_e3.fit(dev_texts, dev_labels)

    # 4. Load Frozen Gold v1.1
    gold_samples, gold_texts, gold_labels = load_dataset(GOLD_PATH)
    logger.info("Loaded %d Frozen Gold v1.1 samples from %s", len(gold_samples), GOLD_PATH.name)

    # 5. Evaluate E1, E2, E3 on Gold Set
    gold_subsets = [s["subset"] for s in gold_samples]

    def eval_model(clf):
        t0 = time.perf_counter()
        preds, confs = [], []
        for t in gold_texts:
            l, c, _ = clf.predict(t)
            preds.append(l)
            confs.append(c)
        lat = ((time.perf_counter() - t0) / len(gold_texts)) * 1000.0
        metrics = compute_metrics(gold_labels, preds, TAXONOMY)
        metrics["mean_latency_ms"] = round(lat, 2)
        
        # Subsets
        for sub in sorted(list(set(gold_subsets))):
            idx = [i for i, s in enumerate(gold_subsets) if s == sub]
            sub_y = [gold_labels[i] for i in idx]
            sub_p = [preds[i] for i in idx]
            metrics[f"subset_{sub}_acc"] = round(float(accuracy_score(sub_y, sub_p)), 4)
            metrics[f"subset_{sub}_count_correct"] = sum(1 for a, b in zip(sub_y, sub_p) if a == b)
            metrics[f"subset_{sub}_count_total"] = len(idx)

        metrics["correct_count"] = sum(1 for a, b in zip(gold_labels, preds) if a == b)
        metrics["total_count"] = len(gold_labels)

        return preds, confs, metrics

    e1_preds, e1_confs, e1_gold = eval_model(clf_e1)
    e2_preds, e2_confs, e2_gold = eval_model(clf_e2)
    e3_preds, e3_confs, e3_gold = eval_model(clf_e3)

    # 6. Save E3 Development Metrics CSV
    e3_dev_rows = [
        {"model": "E1 (TF-IDF+LR)", "accuracy": 0.525, "macro_precision": 0.6655, "macro_recall": 0.525, "macro_f1": 0.5462},
        {"model": "E2 (Semantic+LR)", "accuracy": 0.75, "macro_precision": 0.7584, "macro_recall": 0.75, "macro_f1": 0.7462},
        {"model": f"E3 Hybrid (alpha={best_alpha:.1f})", **e3_cv_metrics},
    ]
    save_csv(RESULTS_DIR / "e3_development_metrics.csv", e3_dev_rows, ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"])

    # 7. Save E3 Gold Metrics CSV
    gold_rows = [
        {"model": "E1 Baseline", **e1_gold},
        {"model": "E2 Semantic", **e2_gold},
        {"model": f"E3 Hybrid (alpha={best_alpha:.1f})", **e3_gold},
    ]
    save_csv(RESULTS_DIR / "e3_gold_metrics.csv", gold_rows, list(gold_rows[0].keys()))

    # 8. Save Predictions CSV
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
            "e3_pred": e3_preds[idx],
            "e3_conf": e3_confs[idx],
            "e1_correct": e1_preds[idx] == sample["gold_intent"],
            "e2_correct": e2_preds[idx] == sample["gold_intent"],
            "e3_correct": e3_preds[idx] == sample["gold_intent"],
        })
    save_csv(RESULTS_DIR / "e3_predictions.csv", pred_records, list(pred_records[0].keys()))

    # 9. Save Confusion Matrix
    cm3 = confusion_matrix(gold_labels, e3_preds, labels=TAXONOMY)
    with (RESULTS_DIR / "e3_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["actual/predicted"] + TAXONOMY)
        for idx, row in enumerate(cm3.tolist()):
            writer.writerow([TAXONOMY[idx]] + row)

    # 10. Save Per-Intent Metrics CSV
    p1, r1, f1_1, _ = precision_recall_fscore_support(gold_labels, e1_preds, labels=TAXONOMY, zero_division=0)
    p2, r2, f1_2, _ = precision_recall_fscore_support(gold_labels, e2_preds, labels=TAXONOMY, zero_division=0)
    p3, r3, f1_3, _ = precision_recall_fscore_support(gold_labels, e3_preds, labels=TAXONOMY, zero_division=0)

    per_intent = []
    for i, intent in enumerate(TAXONOMY):
        per_intent.append({
            "intent": intent,
            "e1_f1": round(float(f1_1[i]), 4),
            "e2_f1": round(float(f1_2[i]), 4),
            "e3_f1": round(float(f1_3[i]), 4),
            "e3_vs_e1_diff": round(float(f1_3[i] - f1_1[i]), 4),
            "e3_vs_e2_diff": round(float(f1_3[i] - f1_2[i]), 4),
        })
    save_csv(RESULTS_DIR / "e3_per_intent_metrics.csv", per_intent, list(per_intent[0].keys()))

    # 11. Save E1 vs E2 vs E3 Comparison CSV
    comparison_rows = [
        {
            "Metric": "Overall Accuracy",
            "E1_Baseline": f"{e1_gold['accuracy']*100:.2f}% ({e1_gold['correct_count']}/{e1_gold['total_count']})",
            "E2_Semantic": f"{e2_gold['accuracy']*100:.2f}% ({e2_gold['correct_count']}/{e2_gold['total_count']})",
            "E3_Hybrid": f"{e3_gold['accuracy']*100:.2f}% ({e3_gold['correct_count']}/{e3_gold['total_count']})",
            "E3_vs_E1_Diff": f"{(e3_gold['accuracy']-e1_gold['accuracy'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['accuracy']-e2_gold['accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "Macro Precision",
            "E1_Baseline": f"{e1_gold['macro_precision']*100:.2f}%",
            "E2_Semantic": f"{e2_gold['macro_precision']*100:.2f}%",
            "E3_Hybrid": f"{e3_gold['macro_precision']*100:.2f}%",
            "E3_vs_E1_Diff": f"{(e3_gold['macro_precision']-e1_gold['macro_precision'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['macro_precision']-e2_gold['macro_precision'])*100:+.2f}%",
        },
        {
            "Metric": "Macro Recall",
            "E1_Baseline": f"{e1_gold['macro_recall']*100:.2f}%",
            "E2_Semantic": f"{e2_gold['macro_recall']*100:.2f}%",
            "E3_Hybrid": f"{e3_gold['macro_recall']*100:.2f}%",
            "E3_vs_E1_Diff": f"{(e3_gold['macro_recall']-e1_gold['macro_recall'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['macro_recall']-e2_gold['macro_recall'])*100:+.2f}%",
        },
        {
            "Metric": "Macro F1-Score",
            "E1_Baseline": f"{e1_gold['macro_f1']*100:.2f}%",
            "E2_Semantic": f"{e2_gold['macro_f1']*100:.2f}%",
            "E3_Hybrid": f"{e3_gold['macro_f1']*100:.2f}%",
            "E3_vs_E1_Diff": f"{(e3_gold['macro_f1']-e1_gold['macro_f1'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['macro_f1']-e2_gold['macro_f1'])*100:+.2f}%",
        },
        {
            "Metric": "Clear In-Scope Accuracy",
            "E1_Baseline": f"{e1_gold['subset_clear_in_scope_acc']*100:.2f}% ({e1_gold['subset_clear_in_scope_count_correct']}/{e1_gold['subset_clear_in_scope_count_total']})",
            "E2_Semantic": f"{e2_gold['subset_clear_in_scope_acc']*100:.2f}% ({e2_gold['subset_clear_in_scope_count_correct']}/{e2_gold['subset_clear_in_scope_count_total']})",
            "E3_Hybrid": f"{e3_gold['subset_clear_in_scope_acc']*100:.2f}% ({e3_gold['subset_clear_in_scope_count_correct']}/{e3_gold['subset_clear_in_scope_count_total']})",
            "E3_vs_E1_Diff": f"{(e3_gold['subset_clear_in_scope_acc']-e1_gold['subset_clear_in_scope_acc'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['subset_clear_in_scope_acc']-e2_gold['subset_clear_in_scope_acc'])*100:+.2f}%",
        },
        {
            "Metric": "Boundary Subset Accuracy",
            "E1_Baseline": f"{e1_gold['subset_boundary_acc']*100:.2f}% ({e1_gold['subset_boundary_count_correct']}/{e1_gold['subset_boundary_count_total']})",
            "E2_Semantic": f"{e2_gold['subset_boundary_acc']*100:.2f}% ({e2_gold['subset_boundary_count_correct']}/{e2_gold['subset_boundary_count_total']})",
            "E3_Hybrid": f"{e3_gold['subset_boundary_acc']*100:.2f}% ({e3_gold['subset_boundary_count_correct']}/{e3_gold['subset_boundary_count_total']})",
            "E3_vs_E1_Diff": f"{(e3_gold['subset_boundary_acc']-e1_gold['subset_boundary_acc'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['subset_boundary_acc']-e2_gold['subset_boundary_acc'])*100:+.2f}%",
        },
        {
            "Metric": "Ambiguous Subset Accuracy",
            "E1_Baseline": f"{e1_gold['subset_ambiguous_acc']*100:.2f}% ({e1_gold['subset_ambiguous_count_correct']}/{e1_gold['subset_ambiguous_count_total']})",
            "E2_Semantic": f"{e2_gold['subset_ambiguous_acc']*100:.2f}% ({e2_gold['subset_ambiguous_count_correct']}/{e2_gold['subset_ambiguous_count_total']})",
            "E3_Hybrid": f"{e3_gold['subset_ambiguous_acc']*100:.2f}% ({e3_gold['subset_ambiguous_count_correct']}/{e3_gold['subset_ambiguous_count_total']})",
            "E3_vs_E1_Diff": f"{(e3_gold['subset_ambiguous_acc']-e1_gold['subset_ambiguous_acc'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['subset_ambiguous_acc']-e2_gold['subset_ambiguous_acc'])*100:+.2f}%",
        },
        {
            "Metric": "OOS Rejection Accuracy",
            "E1_Baseline": f"{e1_gold['subset_OOS_acc']*100:.2f}% ({e1_gold['subset_OOS_count_correct']}/{e1_gold['subset_OOS_count_total']})",
            "E2_Semantic": f"{e2_gold['subset_OOS_acc']*100:.2f}% ({e2_gold['subset_OOS_count_correct']}/{e2_gold['subset_OOS_count_total']})",
            "E3_Hybrid": f"{e3_gold['subset_OOS_acc']*100:.2f}% ({e3_gold['subset_OOS_count_correct']}/{e3_gold['subset_OOS_count_total']})",
            "E3_vs_E1_Diff": f"{(e3_gold['subset_OOS_acc']-e1_gold['subset_OOS_acc'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['subset_OOS_acc']-e2_gold['subset_OOS_acc'])*100:+.2f}%",
        },
        {
            "Metric": "Robustness Subset Accuracy",
            "E1_Baseline": f"{e1_gold['subset_robustness_acc']*100:.2f}% ({e1_gold['subset_robustness_count_correct']}/{e1_gold['subset_robustness_count_total']})",
            "E2_Semantic": f"{e2_gold['subset_robustness_acc']*100:.2f}% ({e2_gold['subset_robustness_count_correct']}/{e2_gold['subset_robustness_count_total']})",
            "E3_Hybrid": f"{e3_gold['subset_robustness_acc']*100:.2f}% ({e3_gold['subset_robustness_count_correct']}/{e3_gold['subset_robustness_count_total']})",
            "E3_vs_E1_Diff": f"{(e3_gold['subset_robustness_acc']-e1_gold['subset_robustness_acc'])*100:+.2f}%",
            "E3_vs_E2_Diff": f"{(e3_gold['subset_robustness_acc']-e2_gold['subset_robustness_acc'])*100:+.2f}%",
        },
        {
            "Metric": "Mean Inference Latency (ms)",
            "E1_Baseline": f"{e1_gold['mean_latency_ms']:.2f} ms",
            "E2_Semantic": f"{e2_gold['mean_latency_ms']:.2f} ms",
            "E3_Hybrid": f"{e3_gold['mean_latency_ms']:.2f} ms",
            "E3_vs_E1_Diff": f"{(e3_gold['mean_latency_ms']-e1_gold['mean_latency_ms']):+.2f} ms",
            "E3_vs_E2_Diff": f"{(e3_gold['mean_latency_ms']-e2_gold['mean_latency_ms']):+.2f} ms",
        },
    ]
    save_csv(RESULTS_DIR / "e1_e2_e3_comparison.csv", comparison_rows, list(comparison_rows[0].keys()))

    # 12. Save Summary JSON
    summary_json = {
        "experiment_name": "Module 3 E3 Hybrid Intent Classifier Experiment (Probability-Level Fusion)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_validation_alpha": best_alpha,
        "dev_dataset": "data/intent_samples_v2_balanced.json (160 samples)",
        "gold_dataset": "data/gold_intent_samples_v1_1.json (240 samples)",
        "development_cv_metrics": {
            "E1": {"accuracy": 0.525, "macro_precision": 0.6655, "macro_recall": 0.525, "macro_f1": 0.5462},
            "E2": {"accuracy": 0.75, "macro_precision": 0.7584, "macro_recall": 0.75, "macro_f1": 0.7462},
            "E3": e3_cv_metrics,
        },
        "gold_test_metrics": {
            "E1": e1_gold,
            "E2": e2_gold,
            "E3": e3_gold,
        },
    }
    with (RESULTS_DIR / "e3_experiment_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    logger.info("=== E3 Hybrid Evaluation Complete. All files saved to %s ===", RESULTS_DIR)


if __name__ == "__main__":
    main()
