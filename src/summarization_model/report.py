"""Console reporting: approach explanations and a results/metrics table."""

from __future__ import annotations

from .metrics import SummaryMetrics

APPROACH_EXPLANATIONS = {
    "Extractive": (
        "Selects and stitches together existing sentences from the source text. "
        "This implementation embeds every sentence with a pretrained Sentence-"
        "Transformer model (all-MiniLM-L6-v2) and ranks sentences by centrality "
        "using TextRank (a PageRank variant over a sentence-similarity graph).\n"
        "  + Factually grounded (never hallucinates), fast, cheap.\n"
        "  - Can read as disjointed, since sentences are lifted verbatim."
    ),
    "Abstractive": (
        "Generates new sentences that paraphrase and compress the source, the "
        "way a human analyst would. This implementation uses a pretrained "
        "sequence-to-sequence transformer (distilbart-cnn-12-6, a distilled "
        "BART model fine-tuned on CNN/DailyMail) via Hugging Face Transformers.\n"
        "  + Fluent and concise; can merge information across sentences.\n"
        "  - Higher compute cost; small risk of imprecise or hallucinated "
        "figures, a real concern when summarizing financial numbers."
    ),
    "Hybrid": (
        "Runs the extractive stage first to narrow the source down to its "
        "most salient sentences, then feeds that condensed text into the same "
        "abstractive transformer to produce the final summary.\n"
        "  + Keeps the generator focused on high-signal content (useful for "
        "long filings that exceed a model's context window) while still "
        "producing fluent, non-verbatim prose.\n"
        "  - Two models to run, so the highest latency of the three; quality "
        "depends on the extractive stage not discarding key sentences."
    ),
}


def print_explanations() -> None:
    print("=" * 88)
    print("SUMMARIZATION APPROACHES")
    print("=" * 88)
    for name, text in APPROACH_EXPLANATIONS.items():
        print(f"\n[{name}]")
        print(text)
    print()


def print_summary(approach: str, text: str) -> None:
    print("-" * 88)
    print(f"{approach.upper()} SUMMARY")
    print("-" * 88)
    print(text)
    print()


def print_metrics_table(all_metrics: list[SummaryMetrics]) -> None:
    headers = [
        "Approach", "Latency(s)", "Words", "Sent.", "Compression",
        "ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1",
    ]
    rows = [
        [
            m.approach,
            f"{m.latency_seconds:.2f}",
            str(m.word_count),
            str(m.sentence_count),
            f"{m.compression_ratio:.1%}",
            f"{m.rouge1_f1:.3f}",
            f"{m.rouge2_f1:.3f}",
            f"{m.rougeL_f1:.3f}",
        ]
        for m in all_metrics
    ]

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print("=" * 88)
    print("KEY EVALUATION METRICS  (ROUGE computed against the human-written reference summary)")
    print("=" * 88)
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in rows:
        print(fmt_row(row))
    print()
    print(
        "ROUGE-1/2 = unigram/bigram overlap with the reference; ROUGE-L = longest\n"
        "common subsequence overlap (rewards matching sentence-level structure).\n"
        "Compression = summary word count / source word count (lower = more concise).\n"
        "Latency = wall-clock time for that approach's forward pass(es) on this machine."
    )
