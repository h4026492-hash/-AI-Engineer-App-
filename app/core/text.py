"""Shared text normalisation.

Lives in ``core`` because both the retrieval layer and the offline model need
the same tokenisation: if they disagree about what a word is, retrieval and
generation silently optimise for different things.

The stemmer is deliberately crude -- strip a few English suffixes, nothing more.
It exists so that "refund" matches "refunds" and "encrypted" matches "encrypts".
A real embedding model handles morphology properly; this keeps the offline path
from failing on trivially related words.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")

# Longest first so "ing" is tried before "s".
_SUFFIXES = ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "s")
_MIN_STEM_LENGTH = 4

# Function words carry no topical signal but are extremely frequent. Left in,
# they dominate similarity: "what is the capital of France" matches almost any
# English sentence on "is" and "the". Dropping them is what makes lexical
# overlap mean anything.
STOP_WORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
    ]
)


def stem(word: str) -> str:
    """Apply light suffix stripping. Never reduces a word below ``_MIN_STEM_LENGTH``."""
    if len(word) <= _MIN_STEM_LENGTH:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix):
            candidate = word[: -len(suffix)]
            if len(candidate) >= _MIN_STEM_LENGTH:
                return candidate
            return word
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, stop-word-free, stemmed tokens in document order."""
    return [stem(word) for word in _WORD.findall(text.lower()) if word not in STOP_WORDS]


def token_set(text: str) -> set[str]:
    """Deduplicated stemmed content tokens, for overlap comparisons."""
    return set(tokenize(text))
