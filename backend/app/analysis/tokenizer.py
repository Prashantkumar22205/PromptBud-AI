"""
analysis/tokenizer.py

Token counting using local tokenizer libraries.
No LLM API is called here.

Design:
  The `BaseTokenizer` abstract class defines the interface.  Concrete
  implementations (`TiktokenTokenizer`) satisfy it.  The `get_tokenizer()`
  factory returns the default implementation.

  Adding a new tokenizer later requires only:
    1. Implement BaseTokenizer.
    2. Register it in get_tokenizer() / TOKENIZER_REGISTRY.
"""

from __future__ import annotations

import abc
import logging
from functools import lru_cache
from typing import Dict, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseTokenizer(abc.ABC):
    """Interface that all tokenizer implementations must satisfy."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'tiktoken/cl100k_base'."""

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for *text*."""


# ---------------------------------------------------------------------------
# tiktoken implementation
# ---------------------------------------------------------------------------


class TiktokenTokenizer(BaseTokenizer):
    """
    Token counter backed by the tiktoken library (OpenAI's tokenizer).

    The default encoding is cl100k_base, which is used by GPT-3.5 and GPT-4.

    IMPORTANT:
        This tokenizer was designed for OpenAI models.  Token counts it
        produces are accurate for GPT-3.5/4 but may differ from the counts
        that Gemini, Claude, or other models would report for the same text.
        This implementation establishes a useful *baseline* count; do not
        treat it as an exact count for non-OpenAI models.
    """

    _ENCODING_NAME = "cl100k_base"

    def __init__(self) -> None:
        try:
            import tiktoken  # local import so the class can be defined without tiktoken installed

            self._enc = tiktoken.get_encoding(self._ENCODING_NAME)
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken is not installed.  Run: pip install tiktoken"
            ) from exc

    @property
    def name(self) -> str:
        return f"tiktoken/{self._ENCODING_NAME}"

    def count_tokens(self, text: str) -> int:
        """Return the number of cl100k_base tokens in *text*."""
        return len(self._enc.encode(text))


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {
    "tiktoken": TiktokenTokenizer,
    # Future entries:
    # "sentencepiece": SentencePieceTokenizer,
    # "huggingface_bert": HuggingFaceBertTokenizer,
}

_DEFAULT_TOKENIZER_KEY = "tiktoken"


@lru_cache(maxsize=4)
def _build_tokenizer(key: str) -> BaseTokenizer:
    """Cache tokenizer instances because initialisation can be expensive."""
    cls = TOKENIZER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown tokenizer key: '{key}'.  Available: {list(TOKENIZER_REGISTRY)}")
    return cls()


def get_tokenizer(key: str = _DEFAULT_TOKENIZER_KEY) -> BaseTokenizer:
    """Return a cached tokenizer instance by registry key."""
    return _build_tokenizer(key)
