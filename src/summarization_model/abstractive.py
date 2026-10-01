"""Abstractive summarization: a pretrained sequence-to-sequence transformer."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


class AbstractiveSummarizer:
    """Generates a new, paraphrased summary rather than copying sentences.

    Uses a pretrained seq2seq transformer (a distilled BART model
    fine-tuned on CNN/DailyMail), loaded directly via the model/tokenizer
    API rather than the high-level `pipeline("summarization", ...)` helper
    -- that task name has moved around across `transformers` major versions,
    while `AutoModelForSeq2SeqLM` + `.generate()` is the stable, version-
    independent interface. The model can merge information across
    sentences and rephrase it, at the cost of a small risk of imprecise or
    hallucinated details -- worth watching closely when the source contains
    financial figures.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if self._model.generation_config.forced_bos_token_id is None:
            self._model.generation_config.forced_bos_token_id = self._model.config.bos_token_id

    def summarize(self, text: str, max_length: int = 140, min_length: int = 40) -> str:
        inputs = self._tokenizer(text, truncation=True, return_tensors="pt")
        output_ids = self._model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            do_sample=False,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
