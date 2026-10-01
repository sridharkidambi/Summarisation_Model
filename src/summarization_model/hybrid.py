"""Hybrid summarization: extractive pre-filtering followed by abstractive generation."""

from __future__ import annotations

from .abstractive import DEFAULT_MODEL as DEFAULT_ABSTRACTIVE_MODEL
from .abstractive import AbstractiveSummarizer
from .extractive import DEFAULT_MODEL as DEFAULT_EXTRACTIVE_MODEL
from .extractive import ExtractiveSummarizer


class HybridSummarizer:
    """Two-stage pipeline: extract the most salient sentences, then abstract.

    Stage 1 (extractive) uses TextRank over pretrained sentence embeddings
    to narrow the source down to its highest-signal sentences -- this keeps
    the generator focused and lets long documents fit within a model's
    context window. Stage 2 (abstractive) feeds that condensed text through
    a pretrained seq2seq transformer to produce a fluent, non-verbatim
    summary. In practice this tends to balance faithfulness and readability
    better than either approach alone.
    """

    def __init__(
        self,
        extractive: ExtractiveSummarizer | None = None,
        abstractive: AbstractiveSummarizer | None = None,
        extractive_model: str = DEFAULT_EXTRACTIVE_MODEL,
        abstractive_model: str = DEFAULT_ABSTRACTIVE_MODEL,
        extractive_sentences: int = 8,
    ) -> None:
        self.extractive = extractive or ExtractiveSummarizer(extractive_model)
        self.abstractive = abstractive or AbstractiveSummarizer(abstractive_model)
        self.extractive_sentences = extractive_sentences

    def summarize(self, text: str, max_length: int = 140, min_length: int = 40) -> str:
        condensed = self.extractive.summarize(text, num_sentences=self.extractive_sentences)
        return self.abstractive.summarize(condensed, max_length=max_length, min_length=min_length)
