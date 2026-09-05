# PromptOptAI — Python FastAPI Backend

> **Module 1: User Prompt** + **Module 2: Local Prompt Analysis** + **Module 3: Semantic Analysis**
>
> Deterministic, local prompt analysis and NLP semantic understanding. No LLM API calls are made anywhere in this backend.

---

## Architecture

```
NEXT.JS FRONTEND (port 3000)
        │
        │  HTTP / REST / JSON
        │
        ▼
PYTHON FASTAPI BACKEND (port 8000)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
 Module 2: Local Analysis            Module 3: Semantic Analysis
 (tiktoken, text stats, repetition)   (TF-IDF + LR, spaCy NLP, Rules)
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
                      ANALYSIS JSON
                           │
                           ▼
                    NEXT.JS FRONTEND
```

### Directory Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI app, CORS, health check
│   ├── __init__.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── analyze.py         # POST /api/analyze (Module 2)
│   │       ├── semantic.py        # POST /api/semantic-analyze (Module 3)
│   │       └── __init__.py
│   │
│   ├── analysis/                  # Module 2 — Local Analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py            # Central orchestrator
│   │   ├── tokenizer.py           # Token counting (tiktoken)
│   │   ├── text_stats.py          # Word/sentence statistics
│   │   ├── repetition.py          # Repetition detection
│   │   ├── structure.py           # Structural signal detection
│   │   └── types.py               # Pydantic models
│   │
│   ├── semantic/                  # Module 3 — Semantic Analysis (Phase 3A)
│   │   ├── __init__.py
│   │   ├── semantic_analyzer.py   # Central orchestrator
│   │   ├── types.py               # Pydantic models
│   │   ├── intent/                # Intent classification (TF-IDF + LogisticRegression)
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py      # BaseIntentClassifier, TfidfLRClassifier
│   │   │   └── training_data.py   # Seed dataset loader
│   │   ├── context/               # Context extraction (spaCy noun chunks + rule engine)
│   │   │   ├── __init__.py
│   │   │   └── extractor.py       # BaseContextExtractor, SpacyContextExtractor
│   │   ├── requirements/          # Requirement extraction (dep parse + verb scan + patterns)
│   │   │   ├── __init__.py
│   │   │   └── extractor.py       # BaseRequirementExtractor, SpacyRequirementExtractor
│   │   ├── constraints/           # Constraint extraction (length, format, style, forbidden, etc.)
│   │   │   ├── __init__.py
│   │   │   └── extractor.py       # BaseConstraintExtractor, RuleConstraintExtractor
│   │   ├── ambiguity/             # Ambiguity & conflict detection
│   │   │   ├── __init__.py
│   │   │   └── detector.py        # BaseAmbiguityDetector, RuleAmbiguityDetector
│   │   └── utils/                 # Pattern dictionaries & keyword maps
│   │       ├── __init__.py
│   │       └── patterns.py
│   │
│   └── core/
│       ├── config.py              # Settings (env vars)
│       └── __init__.py
│
├── data/
│   └── intent_samples.json        # 160 development seed samples across 16 intent classes
│
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_analyzer.py           # Module 2 integration tests
│   ├── test_semantic.py           # Module 3 integration & component tests
│   ├── test_tokenizer.py          # Tokenizer unit tests
│   ├── test_repetition.py         # Repetition unit tests
│   └── test_structure.py          # Structure unit tests
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies (FastAPI, uvicorn, pydantic, tiktoken, spacy, scikit-learn)
pip install -r requirements.txt

# Download spaCy English model for NLP analysis
python -m spacy download en_core_web_sm
```

---

## Running FastAPI

```bash
# From the backend/ directory with the virtual environment active:
uvicorn app.main:app --reload --port 8000
```

The API is available at: `http://localhost:8000`

---

## API Endpoints

### `GET /health`

Health check. Returns HTTP 200 if the server is running.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /api/analyze` (Module 2)

Analyses a prompt locally (statistics, tokens, repetition, structure). No LLM API is called.

**Request:**
```json
{
  "prompt": "Explain binary search to a beginner."
}
```

---

### `POST /api/semantic-analyze` (Module 3)

Performs semantic analysis on a prompt locally (intent, context, requirements, constraints, ambiguities).  
**Module 3 ANALYZES the prompt ONLY. It NEVER rewrites, optimizes, or modifies the prompt.**

**Request:**
```json
{
  "prompt": "Explain bubble sort to a beginner in Python under 200 words."
}
```

**Validation:**
- `prompt` must be a non-empty string
- Leading/trailing whitespace is stripped
- Maximum length: 100,000 characters (configurable via `MAX_PROMPT_LENGTH`)
- Prompt content is never executed as code

**Successful Response (HTTP 200):**
```json
{
  "success": true,
  "analysis": {
    "intent": {
      "primaryIntent": "educational_explanation",
      "confidence": 0.13,
      "method": "tfidf_logistic_regression",
      "secondaryIntents": ["code_explanation"]
    },
    "context": {
      "topic": {
        "value": "bubble sort",
        "confidence": 0.75,
        "method": "spacy_noun_phrase",
        "evidence": "bubble sort"
      },
      "domain": {
        "value": "computer_science",
        "confidence": 0.67,
        "method": "domain_dictionary",
        "evidence": "bubble sort"
      },
      "audience": {
        "value": "beginner",
        "confidence": 0.99,
        "method": "rule",
        "evidence": "to a beginner"
      },
      "programmingLanguage": {
        "value": "Python",
        "confidence": 0.97,
        "method": "rule",
        "evidence": "Python"
      },
      "priorKnowledge": []
    },
    "requirements": [
      {
        "action": "explain",
        "object": "bubble sort",
        "mandatory": true,
        "confidence": 0.9,
        "method": "spacy_dependency_rule",
        "evidence": "Explain bubble sort to a beginner in Python under 200 words."
      }
    ],
    "constraints": [
      {
        "type": "length",
        "description": "Constraint on length: under 200 words",
        "maxWords": 200,
        "minWords": null,
        "outputFormat": null,
        "targetLanguage": null,
        "confidence": 0.96,
        "method": "rule",
        "evidence": "under 200 words"
      }
    ],
    "ambiguities": []
  }
}
```

**Error Responses:**

| HTTP Status | Code | Cause |
|---|---|---|
| 400 | `EMPTY_PROMPT` | Prompt is empty or whitespace-only |
| 413 | `PROMPT_TOO_LARGE` | Prompt exceeds `MAX_PROMPT_LENGTH` |
| 500 | `SEMANTIC_ANALYSIS_ERROR` | Internal error during semantic analysis |

---

## Important Disclaimers

> ⚠️ **Seed Dataset Disclaimer:**  
> The current intent classifier uses a small development seed dataset (`data/intent_samples.json`). It is NOT intended to represent the final research evaluation dataset. It serves as a functional baseline for development and architecture verification.

> ⚠️ **Confidence & Provenance Disclaimer:**  
> Rule-based confidence values are heuristic estimates and classifier probabilities are model confidence estimates, NOT guaranteed measures of semantic correctness. Every extracted field includes a `method` tag for auditability and transparency.

---

## Phase 3A vs Phase 3B Architecture

Module 3 is engineered with clean abstract interfaces (`BaseIntentClassifier`, `BaseContextExtractor`, `BaseRequirementExtractor`, `BaseConstraintExtractor`, `BaseAmbiguityDetector`).

| Component | Phase 3A (Current) | Phase 3B (Future Extension) |
|---|---|---|
| **Intent Classifier** | TF-IDF + LogisticRegression (`TfidfLRClassifier`) | Sentence Transformers / Zero-shot NLI |
| **Context Extractor** | spaCy noun chunks + Keyword Dictionary | Coreference Resolution + Entity Linking |
| **Requirement Extractor** | spaCy Dependency Parse + Verb Patterns | Semantic Role Labeling (SRL) |
| **Constraint Extractor** | Rule-based regex pattern engine | NLI Constraint Entailment |
| **Ambiguity Detector** | Rule-based conflict/underspecification engine | NLI Cross-Encoder Contradiction Detection |

Adding a Phase 3B module requires only implementing the respective abstract base class; zero changes are needed in the core API or orchestrator logic.

---

## Running Tests

```bash
# From the backend/ directory with venv active:
pytest

# With verbose output (runs all 124 unit & integration tests):
pytest -v
```

---

## Swagger / OpenAPI Documentation

Interactive API docs are available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`
