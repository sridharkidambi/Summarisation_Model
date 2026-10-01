"""Evaluation metrics for comparing generated summaries against a reference."""

from __future__ import annotations

from dataclasses import dataclass

from rouge_score import rouge_scorer

from .text_utils import split_sentences

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


@dataclass
class SummaryMetrics:
    approach: str
    latency_seconds: float
    word_count: int
    sentence_count: int
    compression_ratio: float
    rouge1_f1: float
    rouge1_precision: float
    rouge1_recall: float
    rouge2_f1: float
    rougeL_f1: float


def evaluate_summary(
    approach: str,
    summary: str,
    reference: str,
    source_word_count: int,
    latency_seconds: float,
) -> SummaryMetrics:
    """Score a summary against a human-written reference with ROUGE, plus
    simple length/latency metrics that matter in a production setting.

    ROUGE-1 / ROUGE-2 measure unigram / bigram overlap with the reference
    (content coverage); ROUGE-L measures longest-common-subsequence overlap
    (rewards matching the reference's sentence-level structure). Compression
    ratio and latency matter because, outside this demo, a summarizer also
    has to meet a length budget and a performance budget.
    """
    scores = _SCORER.score(reference, summary)
    word_count = len(summary.split())
    return SummaryMetrics(
        approach=approach,
        latency_seconds=latency_seconds,
        word_count=word_count,
        sentence_count=len(split_sentences(summary)),
        compression_ratio=word_count / source_word_count if source_word_count else 0.0,
        rouge1_f1=scores["rouge1"].fmeasure,
        rouge1_precision=scores["rouge1"].precision,
        rouge1_recall=scores["rouge1"].recall,
        rouge2_f1=scores["rouge2"].fmeasure,
        rougeL_f1=scores["rougeL"].fmeasure,
    )
