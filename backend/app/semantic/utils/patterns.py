"""
semantic/utils/patterns.py

Centralised, configurable pattern dictionaries used across Module 3.

Design principles
-----------------
- All pattern data lives here, not scattered across sub-modules.
- Every dictionary/list is a Python constant — easy to extend.
- No LLM API calls anywhere in this file.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Domain keyword dictionary
# ---------------------------------------------------------------------------
# Map domain label → set of trigger keywords.
# To add a new domain: append an entry here; no other code needs changing.

DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "software_engineering": {
        "software", "engineering", "architecture", "design pattern", "microservice",
        "monolith", "deployment", "ci/cd", "devops", "agile", "scrum", "sprint",
        "refactor", "testing", "unit test", "integration test", "api", "rest",
        "graphql", "git", "version control", "pull request", "code review",
        "docker", "kubernetes", "container", "serverless", "cloud",
    },
    "computer_science": {
        "algorithm", "data structure", "complexity", "big-o", "sorting", "searching",
        "binary search", "quicksort", "merge sort", "recursion", "dynamic programming",
        "graph", "tree", "linked list", "stack", "queue", "hash", "heap",
        "compiler", "interpreter", "operating system", "memory", "cpu", "process",
        "thread", "concurrency", "parallelism", "networking", "tcp", "http",
        "database", "sql", "nosql", "index", "query", "join", "normalization",
        "machine learning", "neural network", "deep learning", "ai", "nlp",
        "computer vision", "natural language processing",
    },
    "data_science": {
        "data science", "dataset", "pandas", "numpy", "scipy", "matplotlib",
        "visualization", "regression", "classification", "clustering", "feature",
        "model", "training", "validation", "overfitting", "underfitting",
        "cross-validation", "hyperparameter", "statistics", "probability",
        "distribution", "correlation", "hypothesis", "analysis", "jupyter",
        "notebook", "tensorflow", "pytorch", "keras", "scikit-learn",
    },
    "mathematics": {
        "calculus", "algebra", "geometry", "trigonometry", "linear algebra",
        "matrix", "vector", "derivative", "integral", "differential", "equation",
        "theorem", "proof", "axiom", "topology", "number theory", "combinatorics",
        "probability", "statistics", "arithmetic", "fraction", "logarithm",
        "exponent", "polynomial", "function", "series", "limit",
    },
    "education": {
        "student", "teacher", "lesson", "curriculum", "course", "lecture",
        "assignment", "homework", "grade", "exam", "quiz", "classroom",
        "learning", "teaching", "pedagogy", "education", "school", "university",
        "textbook", "tutorial", "beginner", "novice", "learner",
    },
    "business": {
        "business", "company", "enterprise", "strategy", "management", "revenue",
        "profit", "cost", "market", "customer", "product", "service", "sales",
        "marketing", "brand", "campaign", "stakeholder", "roi", "kpi",
        "startup", "growth", "investment", "budget", "forecast", "report",
    },
    "finance": {
        "finance", "financial", "investment", "stock", "bond", "portfolio",
        "risk", "return", "asset", "liability", "equity", "debt", "capital",
        "market", "trading", "hedge", "dividend", "interest", "loan",
        "mortgage", "tax", "accounting", "audit", "balance sheet", "cash flow",
    },
    "healthcare": {
        "health", "medical", "clinical", "patient", "diagnosis", "treatment",
        "disease", "symptom", "drug", "medication", "hospital", "doctor",
        "nurse", "therapy", "surgery", "research", "trial", "epidemiology",
        "public health", "wellness", "nutrition", "mental health",
    },
    "science": {
        "physics", "chemistry", "biology", "astronomy", "geology", "ecology",
        "experiment", "hypothesis", "research", "laboratory", "observation",
        "theory", "scientific", "molecule", "atom", "element", "cell",
        "evolution", "genetics", "quantum", "relativity",
    },
    "marketing": {
        "marketing", "advertisement", "campaign", "brand", "audience", "content",
        "engagement", "conversion", "funnel", "lead", "seo", "social media",
        "email", "copywriting", "messaging", "positioning", "persona",
        "analytics", "impression", "click-through", "cta",
    },
}

# ---------------------------------------------------------------------------
# Audience patterns
# ---------------------------------------------------------------------------

AUDIENCE_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    # (label, compiled_pattern, confidence)
    ("beginner",       re.compile(r"\b(beginner|novice|new to|just starting|no experience|never (studied|learned|used)|first time)\b", re.I), 0.98),
    ("beginner",       re.compile(r"\b(for a beginner|to a beginner)\b", re.I), 0.99),
    ("expert",         re.compile(r"\b(expert|advanced|senior|experienced|professional|specialist)\b", re.I), 0.95),
    ("intermediate",   re.compile(r"\b(intermediate|some experience|familiar with|basic knowledge)\b", re.I), 0.90),
    ("student",        re.compile(r"\b(student|learner|pupil|undergraduate|graduate)\b", re.I), 0.92),
    ("developer",      re.compile(r"\b(developer|programmer|engineer|coder|software dev)\b", re.I), 0.90),
    ("child",          re.compile(r"\b(child|kid|5.year.old|ten.year.old|young|elementary|primary school)\b", re.I), 0.93),
    ("manager",        re.compile(r"\b(manager|executive|cto|ceo|director|non.technical|business)\b", re.I), 0.85),
    ("researcher",     re.compile(r"\b(researcher|scientist|academic|phd|professor|faculty)\b", re.I), 0.90),
    ("general_public", re.compile(r"\b(general public|layperson|non.expert|everyday|anyone)\b", re.I), 0.88),
]

# ---------------------------------------------------------------------------
# Programming language patterns
# ---------------------------------------------------------------------------

LANGUAGE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Python",     re.compile(r"\b(python|\.py)\b", re.I)),
    ("JavaScript", re.compile(r"\b(javascript|js|node\.?js|nodejs|es6|ecmascript)\b", re.I)),
    ("TypeScript", re.compile(r"\b(typescript|\.tsx?)\b", re.I)),
    ("Java",       re.compile(r"\bjava(?!script)\b", re.I)),
    ("C++",        re.compile(r"\bc\+\+|cpp\b", re.I)),
    ("C",          re.compile(r"\bthe\s+c\s+language\b|\bc\s+programming\b|\b\.c\b", re.I)),
    ("Go",         re.compile(r"\b(golang|go\s+language|\bgo\b\s+(code|function|program))\b", re.I)),
    ("Rust",       re.compile(r"\brust\b", re.I)),
    ("SQL",        re.compile(r"\b(sql|mysql|postgresql|sqlite|t-sql|plsql)\b", re.I)),
    ("R",          re.compile(r"\br\s+(language|programming|code)\b|\bggplot\b|\btidyverse\b", re.I)),
    ("Swift",      re.compile(r"\bswift\b|\bswiftui\b", re.I)),
    ("Kotlin",     re.compile(r"\bkotlin\b", re.I)),
    ("Ruby",       re.compile(r"\bruby\b|\brails\b|\bruby on rails\b", re.I)),
    ("PHP",        re.compile(r"\bphp\b|\blaravel\b|\bwordpress\b", re.I)),
    ("C#",         re.compile(r"\bc#|csharp|\.net|dotnet\b", re.I)),
    ("Bash",       re.compile(r"\b(bash|shell script|sh|zsh|powershell)\b", re.I)),
    ("Scala",      re.compile(r"\bscala\b", re.I)),
    ("Haskell",    re.compile(r"\bhaskell\b", re.I)),
]

# ---------------------------------------------------------------------------
# Prior knowledge patterns
# ---------------------------------------------------------------------------

PRIOR_KNOWLEDGE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("assumes_knowledge",  re.compile(r"\bwho\s+knows?\s+(.{3,60}?)(?:\.|,|and|but)\b", re.I)),
    ("assumes_knowledge",  re.compile(r"\bassume\s+(knowledge|familiarity|understanding)\s+of\s+(.{3,60}?)(?:\.|,)\b", re.I)),
    ("assumes_knowledge",  re.compile(r"\bfamiliar\s+with\s+(.{3,60}?)(?:\.|,|and|but)\b", re.I)),
    ("no_assumption",      re.compile(r"\bno\s+(prior\s+)?knowledge\s+of\b", re.I)),
    ("no_assumption",      re.compile(r"\bwithout\s+assuming\b", re.I)),
    ("no_assumption",      re.compile(r"\bfrom\s+scratch\b|\bfrom\s+the\s+beginning\b|\bfrom\s+basics\b", re.I)),
]

# ---------------------------------------------------------------------------
# Requirement action verbs
# ---------------------------------------------------------------------------

REQUIREMENT_VERBS: Set[str] = {
    "explain", "describe", "provide", "generate", "write", "compare", "summarize",
    "summarise", "translate", "analyze", "analyse", "calculate", "demonstrate",
    "show", "include", "list", "debug", "implement", "create", "build", "design",
    "outline", "define", "evaluate", "give", "teach", "illustrate", "perform",
    "compute", "solve", "find", "identify", "extract", "convert", "format",
    "produce", "develop", "make", "draw", "display", "walk", "cover", "address",
}

# ---------------------------------------------------------------------------
# Constraint patterns  (type, compiled_pattern, default_confidence)
# ---------------------------------------------------------------------------

CONSTRAINT_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    # LENGTH
    ("length", re.compile(r"\b(under|within|less than|at most|no more than|maximum of?|max)\s+\d+\s*(words?|sentences?|characters?|lines?|paragraphs?)\b", re.I), 0.96),
    ("length", re.compile(r"\bkeep\s+(it|the\s+answer|the\s+response)\s+(short|brief|concise|to\s+the\s+point)\b", re.I), 0.90),
    ("length", re.compile(r"\b(short|brief|concise)\s+(answer|response|explanation|summary)\b", re.I), 0.85),
    ("length", re.compile(r"\bno\s+longer\s+than\s+\d+\b", re.I), 0.95),
    ("length", re.compile(r"\blimit\s+(the\s+)?(answer|response|explanation)\s+to\b", re.I), 0.93),

    # FORMAT
    ("format", re.compile(r"\b(use|in|as)\s+(a\s+)?bullet(ed)?\s*(point|list)?s?\b", re.I), 0.97),
    ("format", re.compile(r"\buse\s+(numbered|ordered)\s+(list|steps?)\b", re.I), 0.97),
    ("format", re.compile(r"\b(use|with)\s+headings?\b", re.I), 0.95),
    ("format", re.compile(r"\breturn\s+(as\s+)?json\b", re.I), 0.97),
    ("format", re.compile(r"\bformat\s+(as|it|the\s+response)\b", re.I), 0.90),
    ("format", re.compile(r"\bin\s+(json|xml|csv|markdown|html|yaml)\s*(format|form)?\b", re.I), 0.95),
    ("format", re.compile(r"\bprovide\s+a\s+table\b", re.I), 0.95),
    ("format", re.compile(r"\bas\s+a\s+(list|table|chart|diagram|outline)\b", re.I), 0.88),
    ("format", re.compile(r"\bstep.by.step\b", re.I), 0.85),

    # STYLE
    ("style", re.compile(r"\b(in|use)\s+simple\s+(language|terms)\b", re.I), 0.96),
    ("style", re.compile(r"\buse\s+plain\s+language\b", re.I), 0.95),
    ("style", re.compile(r"\bsimple\s+(and\s+)?(clear|easy)\b", re.I), 0.88),
    ("style", re.compile(r"\bformal\s+(tone|language|style)\b", re.I), 0.92),
    ("style", re.compile(r"\binformal\s+(tone|language|style)\b", re.I), 0.92),
    ("style", re.compile(r"\bconversational\b", re.I), 0.88),
    ("style", re.compile(r"\bprofessional\s+(tone|language|style)\b", re.I), 0.90),
    ("style", re.compile(r"\bavoid\s+(jargon|technical\s+terms|complex\s+words)\b", re.I), 0.93),

    # LANGUAGE
    ("language", re.compile(r"\b(answer|respond|write|translate)\s+(in|into)\s+(hindi|french|german|spanish|chinese|japanese|arabic|portuguese|russian|italian|korean|dutch|turkish|polish|swedish|danish|norwegian|finnish|greek|hebrew|thai|vietnamese|malay|indonesian|bengali|urdu|persian|farsi)\b", re.I), 0.97),
    ("language", re.compile(r"\b(in|into)\s+(hindi|french|german|spanish|chinese|japanese|arabic|portuguese|russian|italian|korean)\b", re.I), 0.92),

    # TECHNOLOGY
    ("technology", re.compile(r"\b(use|using|with|in)\s+(python|javascript|java|c\+\+|typescript|go|rust|sql|r)\b", re.I), 0.88),
    ("technology", re.compile(r"\bpython\s+(code|example|implementation|function)\b", re.I), 0.93),
    ("technology", re.compile(r"\bjavascript\s+(code|example|implementation|function)\b", re.I), 0.93),

    # FORBIDDEN
    ("forbidden", re.compile(r"\bdo\s+not\s+use\s+(.{3,60}?)(?:\.|$|,|\n)", re.I), 0.96),
    ("forbidden", re.compile(r"\bdon'?t\s+use\s+(.{3,60}?)(?:\.|$|,|\n)", re.I), 0.95),
    ("forbidden", re.compile(r"\bwithout\s+using\s+(.{3,60}?)(?:\.|$|,|\n)", re.I), 0.93),
    ("forbidden", re.compile(r"\bno\s+external\s+librar", re.I), 0.96),
    ("forbidden", re.compile(r"\bdo\s+not\s+repeat\b", re.I), 0.92),
    ("forbidden", re.compile(r"\bavoid\s+using\b", re.I), 0.90),

    # SCOPE
    ("scope", re.compile(r"\bonly\s+(discuss|cover|focus\s+on|explain|address)\b", re.I), 0.91),
    ("scope", re.compile(r"\bstick\s+to\b", re.I), 0.88),
    ("scope", re.compile(r"\bdo\s+not\s+(go\s+beyond|include|discuss)\b", re.I), 0.90),
    ("scope", re.compile(r"\blimit\s+(the\s+)?(scope|discussion|coverage)\s+to\b", re.I), 0.92),

    # REQUIRED / MUST
    ("required", re.compile(r"\bmust\s+include\b", re.I), 0.94),
    ("required", re.compile(r"\bmake\s+sure\b", re.I), 0.88),
    ("required", re.compile(r"\bensure\s+that\b", re.I), 0.88),
    ("required", re.compile(r"\byou\s+must\b", re.I), 0.92),
    ("required", re.compile(r"\bplease\s+include\b", re.I), 0.85),
    ("required", re.compile(r"\bshould\s+include\b", re.I), 0.82),
]

# ---------------------------------------------------------------------------
# Ambiguity: conflicting constraint pairs  (label_a, label_b, pattern_a, pattern_b)
# ---------------------------------------------------------------------------

CONFLICTING_CONSTRAINT_PAIRS: List[Tuple[str, str, re.Pattern, re.Pattern]] = [
    (
        "keep_short",
        "explain_detail",
        re.compile(r"\b(very\s+short|keep\s+(it\s+)?short|brief|concise|to\s+the\s+point|under\s+\d+\s*words?)\b", re.I),
        re.compile(r"\b(exhaustive\s+detail|every\s+step\s+in\s+detail|in\s+depth|thoroughly|comprehensive|detailed\s+explanation|explain\s+each)\b", re.I),
    ),
    (
        "simple_language",
        "technical_depth",
        re.compile(r"\b(simple\s+language|plain\s+language|easy\s+to\s+understand|non.technical|avoid\s+jargon)\b", re.I),
        re.compile(r"\b(technical\s+detail|in.depth\s+technical|advanced\s+concept|internal\s+mechanism|low.level)\b", re.I),
    ),
    (
        "no_examples",
        "with_examples",
        re.compile(r"\bdo\s+not\s+(give|provide|include|use)\s+(any\s+)?examples?\b", re.I),
        re.compile(r"\bgive\s+(a\s+)?(code\s+)?example|include\s+(a\s+)?example|provide\s+(a\s+)?example|with\s+example\b", re.I),
    ),
    (
        "formal_tone",
        "informal_tone",
        re.compile(r"\bformal\s+(tone|language|style)\b", re.I),
        re.compile(r"\binformal|casual|conversational|friendly\s+tone\b", re.I),
    ),
]

# ---------------------------------------------------------------------------
# Underspecification patterns
# ---------------------------------------------------------------------------

UNDERSPECIFIED_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    ("underspecified_object", re.compile(r"\b(fix|fix\s+this|fix\s+it|fix\s+the\s+code)\b", re.I), "Object of 'fix' is not specified"),
    ("underspecified_object", re.compile(r"\b(explain\s+this|explain\s+it)\b", re.I), "Object of 'explain' is not specified"),
    ("underspecified_object", re.compile(r"\b(make\s+it\s+better|improve\s+this|improve\s+it)\b", re.I), "What to improve is not specified"),
    ("underspecified_object", re.compile(r"\b(do\s+something|help\s+me|process\s+this)\b", re.I), "No specific task is defined"),
    ("missing_context",       re.compile(r"\b(this|that|it|the\s+code|the\s+function|the\s+document)\b(?!.{0,80}(the\s+following|below|above|:\s*\n))", re.I), "Referential term without clear antecedent"),
]

