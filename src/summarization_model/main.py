"""CLI entry point: run extractive, abstractive and hybrid summarization over
a sample financial article and report ROUGE-based evaluation metrics.
"""

from __future__ import annotations

import argparse
import time

from .abstractive import DEFAULT_MODEL as DEFAULT_ABSTRACTIVE_MODEL
from .abstractive import AbstractiveSummarizer
from .extractive import DEFAULT_MODEL as DEFAULT_EXTRACTIVE_MODEL
from .extractive import ExtractiveSummarizer
from .hybrid import HybridSummarizer
from .metrics import evaluate_summary
from .report import print_explanations, print_metrics_table, print_summary
from .sample_text import FINANCIAL_TEXT, REFERENCE_SUMMARY
from .text_utils import word_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extractive-model",
        default=DEFAULT_EXTRACTIVE_MODEL,
        help="Sentence-embedding model used to rank sentences (extractive + hybrid stages).",
    )
    parser.add_argument(
        "--abstractive-model",
        default=DEFAULT_ABSTRACTIVE_MODEL,
        help="Pretrained seq2seq model used for generation (abstractive + hybrid stages).",
    )
    parser.add_argument(
        "--extractive-sentences", type=int, default=6,
        help="Number of sentences the extractive stage keeps.",
    )
    parser.add_argument(
        "--hybrid-sentences", type=int, default=8,
        help="Number of sentences the hybrid stage's extractive pre-filter keeps.",
    )
    parser.add_argument("--max-length", type=int, default=140, help="Max token length for generated output.")
    parser.add_argument("--min-length", type=int, default=40, help="Min token length for generated output.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print_explanations()

    print("SOURCE TEXT (sample financial article)")
    print("-" * 88)
    print(FINANCIAL_TEXT)
    source_words = word_count(FINANCIAL_TEXT)
    print(f"\n(Source length: {source_words} words)\n")

    print("REFERENCE (HUMAN-WRITTEN) SUMMARY -- used as the ROUGE ground truth")
    print("-" * 88)
    print(REFERENCE_SUMMARY)
    print()

    print("Loading pretrained models (first run downloads them; cached afterward) ...")
    extractive = ExtractiveSummarizer(args.extractive_model)
    abstractive = AbstractiveSummarizer(args.abstractive_model)
    hybrid = HybridSummarizer(
        extractive=extractive,
        abstractive=abstractive,
        extractive_sentences=args.hybrid_sentences,
    )
    print("Models loaded.\n")

    all_metrics = []

    start = time.perf_counter()
    extractive_summary = extractive.summarize(FINANCIAL_TEXT, num_sentences=args.extractive_sentences)
    elapsed = time.perf_counter() - start
    print_summary("Extractive", extractive_summary)
    all_metrics.append(
        evaluate_summary("Extractive", extractive_summary, REFERENCE_SUMMARY, source_words, elapsed)
    )

    start = time.perf_counter()
    abstractive_summary = abstractive.summarize(
        FINANCIAL_TEXT, max_length=args.max_length, min_length=args.min_length
    )
    elapsed = time.perf_counter() - start
    print_summary("Abstractive", abstractive_summary)
    all_metrics.append(
        evaluate_summary("Abstractive", abstractive_summary, REFERENCE_SUMMARY, source_words, elapsed)
    )

    start = time.perf_counter()
    hybrid_summary = hybrid.summarize(FINANCIAL_TEXT, max_length=args.max_length, min_length=args.min_length)
    elapsed = time.perf_counter() - start
    print_summary("Hybrid", hybrid_summary)
    all_metrics.append(evaluate_summary("Hybrid", hybrid_summary, REFERENCE_SUMMARY, source_words, elapsed))

    print_metrics_table(all_metrics)


if __name__ == "__main__":
    main()
