"""Extractive summarization: pretrained sentence embeddings + TextRank."""

from __future__ import annotations

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from .text_utils import split_sentences

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ExtractiveSummarizer:
    """Selects the most salient sentences, verbatim, from the source text.

    No new text is generated. A pretrained Sentence-Transformer model
    embeds every sentence; cosine similarity between embeddings forms a
    sentence graph, and TextRank (PageRank over that graph) scores each
    sentence by how central it is to the document. The top-scoring
    sentences are then returned in their original order.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def summarize(self, text: str, num_sentences: int = 5) -> str:
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        embeddings = self._model.encode(sentences, normalize_embeddings=True)
        similarity = np.clip(embeddings @ embeddings.T, 0.0, 1.0)
        np.fill_diagonal(similarity, 0.0)

        graph = nx.from_numpy_array(similarity)
        scores = nx.pagerank(graph)

        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:num_sentences])
        return " ".join(sentences[i] for i in top_indices)
