# Summarization Model Comparison

Compares three text-summarization approaches — **Extractive**, **Abstractive**,
and **Hybrid** — over a sample financial article, using pretrained models for
each, and reports key evaluation metrics (ROUGE-1/2/L, compression ratio,
latency) against a human-written reference summary.

| Approach    | Pretrained model(s)                                              | How it works |
|-------------|-------------------------------------------------------------------|--------------|
| Extractive  | `sentence-transformers/all-MiniLM-L6-v2`                          | Embeds each sentence, ranks them with TextRank (PageRank over a similarity graph), returns the top sentences verbatim. |
| Abstractive | `sshleifer/distilbart-cnn-12-6` (distilled BART, via 🤗 Transformers) | Generates a new, paraphrased summary end-to-end. |
| Hybrid      | Both of the above                                                 | Extractive stage condenses the source, then the abstractive model summarizes the condensed text. |

## Run it

From the repo root (uv workspace):

```bash
uv run --package summarization-model summarization-model
```

or from this directory:

```bash
cd summarization_model
uv sync
uv run summarization-model
```

The first run downloads the two pretrained models (~90 MB + ~300 MB) from
Hugging Face; they're cached locally afterward.

### Useful flags

```bash
uv run summarization-model --extractive-sentences 5 --hybrid-sentences 8 --max-length 120
uv run summarization-model --abstractive-model facebook/bart-large-cnn  # swap in a larger model
```

## What gets printed

1. A plain-language explanation of each approach, with pros/cons.
2. The sample financial article and a human-written reference summary.
3. Each approach's generated summary.
4. A metrics table: latency, word/sentence count, compression ratio, and
   ROUGE-1/2/L F1 scores against the reference summary.

## Metrics, briefly

- **ROUGE-1 / ROUGE-2**: unigram / bigram overlap with the reference summary —
  a proxy for content coverage.
- **ROUGE-L**: longest-common-subsequence overlap — rewards matching the
  reference's sentence-level structure, not just shared words.
- **Compression ratio**: summary word count ÷ source word count.
- **Latency**: wall-clock seconds for that approach's forward pass(es).

ROUGE is a lexical-overlap metric, so the abstractive/hybrid summaries can
score lower than extractive ones even when they read better — abstractive
summaries paraphrase, so they legitimately use different words than the
reference. That tradeoff is part of the point of this comparison.
