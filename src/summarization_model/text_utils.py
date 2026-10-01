"""Small text-processing helpers shared across summarizers."""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter.

    A full NLP tokenizer (e.g. NLTK's punkt) would need its own model
    download; splitting only on sentence-ending punctuation followed by a
    capitalized word is accurate enough for this demo and conveniently
    avoids false splits on decimal numbers ("2.4 billion") and the
    mid-sentence abbreviations ("Inc.", "Corp.") common in financial text,
    since the word right after those is lowercase.
    """
    normalized = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def word_count(text: str) -> int:
    return len(text.split())
