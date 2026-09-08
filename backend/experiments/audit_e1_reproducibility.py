"""
experiments/audit_e1_reproducibility.py

Deep audit of E1 results against previous baseline claims (76.67% accuracy / 81.24% Macro-F1).
"""

import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

V1_PATH = DATA_DIR / "intent_samples.json"
V2_PATH = DATA_DIR / "intent_samples_v2_balanced.json"
GOLD_PATH = DATA_DIR / "gold_intent_samples_v1_1.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["samples"]

v1_samples = load_json(V1_PATH)
v2_samples = load_json(V2_PATH)
gold_samples = load_json(GOLD_PATH)

TAXONOMY_16 = [
    "educational_explanation", "question_answering", "code_generation",
    "code_debugging", "code_explanation", "summarization", "translation",
    "rewriting", "comparison", "data_analysis", "creative_writing",
    "classification", "information_extraction", "problem_solving",
    "content_generation", "other"
]
TAXONOMY_15 = [c for c in TAXONOMY_16 if c != "other"]

train_datasets = [("V1", v1_samples), ("V2", v2_samples)]

# Test subsets
subsets_map = {
    "All 240 Gold": gold_samples,
    "180 In-Scope (Clear+Boundary+Robustness)": [s for s in gold_samples if s["subset"] in ("clear_in_scope", "boundary", "robustness")],
    "172 (Clear+Boundary)": [s for s in gold_samples if s["subset"] in ("clear_in_scope", "boundary")],
    "112 Clear": [s for s in gold_samples if s["subset"] == "clear_in_scope"],
    "210 Excl OOS": [s for s in gold_samples if s["subset"] != "OOS"],
    "210 Excl Ambiguous": [s for s in gold_samples if s["subset"] != "ambiguous"],
}

print(f"{'TrainData':<9} | {'Subset':<32} | {'Thresh':<6} | {'C':<4} | {'Acc (%)':<8} | {'Macro-F1 (16)':<14} | {'Macro-F1 (15)':<14} | {'Weighted-F1':<12}")
print("-" * 120)

for td_name, td_samples in train_datasets:
    train_x = [s["text"] for s in td_samples]
    train_y = [s.get("intent") or s.get("gold_intent") for s in td_samples]
    
    for C in [0.5, 1.0, 2.0, 5.0, 10.0]:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=10000, strip_accents="unicode")),
            ("clf", LogisticRegression(max_iter=1000, C=C, solver="lbfgs", random_state=42))
        ])
        pipe.fit(train_x, train_y)
        classes = list(pipe.classes_)
        
        for sub_name, sub_samples in subsets_map.items():
            g_x = [s["text"] for s in sub_samples]
            g_y = [s["gold_intent"] for s in sub_samples]
            probs = pipe.predict_proba(g_x)
            
            for thresh in [0.0, 0.09]:
                if thresh == 0.0:
                    preds = [classes[int(r.argmax())] for r in probs]
                else:
                    preds = [classes[int(r.argmax())] if r.max() >= thresh else "other" for r in probs]
                
                acc = accuracy_score(g_y, preds)
                _, _, f1_16, _ = precision_recall_fscore_support(g_y, preds, labels=TAXONOMY_16, average="macro", zero_division=0)
                _, _, f1_15, _ = precision_recall_fscore_support(g_y, preds, labels=TAXONOMY_15, average="macro", zero_division=0)
                _, _, f1_w, _ = precision_recall_fscore_support(g_y, preds, labels=TAXONOMY_16, average="weighted", zero_division=0)
                
                # Check for exact match or near match to 76.67% acc / 81.24% F1
                match = ""
                if abs(acc - 0.7667) < 0.01 or abs(f1_16 - 0.8124) < 0.01 or abs(f1_15 - 0.8124) < 0.01:
                    match = " <-- MATCH CANDIDATE!"
                
                print(f"{td_name:<9} | {sub_name:<32} | {thresh:<6} | {C:<4} | {acc*100:6.2f}%  | {f1_16*100:12.2f}% | {f1_15*100:12.2f}% | {f1_w*100:10.2f}%{match}")

