# PromptOptAI — Python FastAPI Backend

> **Module 1: User Prompt** + **Module 2: Local Prompt Analysis**
>
> Deterministic, local prompt analysis. No LLM API calls are made anywhere in this backend for these modules.

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
        ▼
 PROMPT ANALYSIS ENGINE
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Token  Text   NLP
izer   Stats  Rules
 │      │      │
 └──────┼──────┘
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
│   │       ├── analyze.py         # POST /api/analyze
│   │       └── __init__.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analyzer.py            # Central orchestrator
│   │   ├── tokenizer.py           # Token counting (tiktoken)
│   │   ├── text_stats.py          # Word/sentence statistics
│   │   ├── repetition.py          # Repetition detection
│   │   ├── structure.py           # Structural signal detection
│   │   └── types.py               # Pydantic models
│   │
│   └── core/
│       ├── config.py              # Settings (env vars)
│       └── __init__.py
│
├── tests/
│   ├── conftest.py                # Pytest fixtures
│   ├── test_analyzer.py           # Integration tests
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

# Install dependencies
pip install -r requirements.txt
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

### `POST /api/analyze`

Analyses a prompt locally. No LLM API is called.

**Request:**
```json
{
  "prompt": "Explain binary search to a beginner."
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
    "textStats": {
      "characterCount": 123,
      "wordCount": 20,
      "uniqueWordCount": 18,
      "sentenceCount": 3,
      "lineCount": 1,
      "averageWordLength": 5.4,
      "averageSentenceLength": 6.7,
      "minSentenceLength": 4,
      "maxSentenceLength": 9
    },
    "tokenStats": {
      "tokenCount": 28,
      "tokenizer": "tiktoken/cl100k_base"
    },
    "repetition": {
      "repeatedWords": [],
      "repeatedPhrases": [],
      "repetitionScore": 0.0
    },
    "structure": {
      "hasRoleInstruction": false,
      "hasContext": true,
      "hasTaskInstruction": true,
      "hasOutputFormat": false,
      "hasConstraints": false,
      "hasExamples": false,
      "hasQuestion": false,
      "sections": ["Context / Audience", "Task Instruction"]
    }
  }
}
```

**Error Responses:**

| HTTP Status | Code | Cause |
|---|---|---|
| 400 | `EMPTY_PROMPT` | Prompt is empty or whitespace-only |
| 413 | `PROMPT_TOO_LARGE` | Prompt exceeds `MAX_PROMPT_LENGTH` |
| 500 | `ANALYSIS_ERROR` | Internal error during analysis |

---

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `FRONTEND_URL` | `http://localhost:3000` | Allowed CORS origin (Next.js dev server) |
| `MAX_PROMPT_LENGTH` | `100000` | Maximum prompt size in characters |
| `DEFAULT_TOKENIZER` | `tiktoken` | Tokenizer registry key |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `DEBUG` | `false` | FastAPI debug mode |

---

## Local Analysis Methodology

> **Important:** Local Analysis performs deterministic/local analysis and does not attempt to fully understand the semantic intent of the prompt. Detection is structural and heuristic in nature.

### 1. Token Count

Token counting is performed locally using the **tiktoken** library (OpenAI's tokenizer).

The default encoding is `cl100k_base`, used by GPT-3.5 and GPT-4.

> **Important:** Token counts are tokenizer/model dependent. The count produced by this tokenizer is accurate for OpenAI GPT-3.5/4 models but may differ from counts produced by Gemini, Claude, or other models for the same text. This implementation provides a consistent, reproducible baseline count.

The tokenizer interface (`BaseTokenizer`) is designed for extensibility. Adding a new tokenizer requires only implementing `BaseTokenizer` and registering it in `TOKENIZER_REGISTRY`.

### 2. Word & Sentence Statistics

Word extraction uses regex `\b[a-zA-Z0-9']+\b` to handle punctuation correctly. "search," and "search" are treated as the same word.

Sentence splitting uses a rule-based approach that:
- Treats `.`, `?`, `!` as sentence terminators
- Avoids splitting on known abbreviations (Mr., Dr., e.g., etc.)
- Is more accurate than naive period-counting

### 3. Repetition Detection

**Methodology:**

1. Tokenise all words (lowercase, punctuation stripped)
2. Remove stopwords (common function words)
3. Count word frequencies → detect repeated meaningful words
4. Build 2-grams and 3-grams from all tokens
5. Count n-gram frequencies → detect repeated phrases

**Repetition Score Formula:**
```
Let M  = count of meaningful tokens (stopwords excluded)
Let RW = Σ (count − 1) for each repeated meaningful word
Let RP = Σ (count − 1) for each repeated n-gram (2- and 3-word)

repetitionScore = min(1.0, (RW + RP) / max(1, M))
```

- Score of 0.0 = no repetition
- Score approaching 1.0 = extensive repetition
- Values above ~0.3 typically indicate noticeable redundancy
- The formula is deterministic and fully reproducible from the input text

### 4. Structural Analysis

Signal detection uses regex patterns and keyword matching. Seven structural signals are detected:

| Signal | Detection Approach |
|---|---|
| **Role Instruction** | Patterns: "you are a...", "act as...", "pretend to be..." |
| **Context / Audience** | Patterns: "for a beginner", "for students", "target audience..." |
| **Task Instruction** | Action verbs: explain, write, generate, create, summarize, ... |
| **Output Format** | Patterns: "use bullet points", "return JSON", "use headings", ... |
| **Constraints** | Patterns: "do not", "don't", "keep it short", "limit the...", ... |
| **Examples** | Patterns: "for example", "e.g.", "such as", code blocks (```), ... |
| **Question** | Any sentence ending with `?` |

---

## Running Tests

```bash
# From the backend/ directory with venv active:
pytest

# With verbose output:
pytest -v

# With coverage:
pytest --cov=app
```

---

## Swagger / OpenAPI Documentation

Interactive API docs are available at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Limitations

1. **Token count is tokenizer-dependent.** The cl100k_base tokenizer is designed for OpenAI models. Counts may differ for Gemini, Claude, or other providers.
2. **Structural analysis is heuristic.** Pattern-based detection may miss unconventional phrasings or produce false positives.
3. **No semantic understanding.** This module does not interpret the meaning of the prompt — only its surface structure.
4. **Sentence splitter is rule-based.** Complex or unusual punctuation patterns may occasionally be split incorrectly.
5. **No persistence.** Analysis results are computed in memory and not stored.

---

## Future Modules (Not Yet Implemented)

### Module 3 — Semantic Analysis
Planned technologies: `sentence-transformers`, Hugging Face Transformers, NLI models, local classifiers.
Planned features: intent detection, deep context extraction, audience detection, requirement extraction, ambiguity detection.

### Module 4 — Optimization LLM
Will call external LLM APIs (Gemini, OpenAI, Anthropic) to generate an optimized version of the prompt.

### Module 5 — Evaluation
Metrics: token reduction, semantic similarity, requirement preservation, response quality, latency, cost.

### Module 6 — Composite Scoring

### Module 7 — Multi-LLM Benchmarking

---

## Frontend Connection

The Next.js frontend can call this backend at:

```
POST http://localhost:8000/api/analyze
```

The existing Next.js `/api/optimize` route is **not affected** by this backend.
