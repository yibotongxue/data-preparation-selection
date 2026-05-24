from __future__ import annotations

import math

import pandas as pd
from dataflow.operators.eval import PerplexityScorer

from data_selection.utils import extract_text


class PerplexityBasedSelector:
    """Select samples by perplexity using DataFlow's Kenlm-based scorer.

    Supports three strategies:
      - "low": lowest PPL (most fluent)
      - "high": highest PPL (most difficult)
      - "mid": PPL closest to mean (sweet spot)

    Requires Kenlm model files downloaded from HuggingFace.
    """

    def __init__(
        self,
        k: int = 100,
        strategy: str = "low",
        text_key: str = "text",
        lang: str = "en",
        model_name: str = "dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
        scorer: PerplexityScorer | None = None,
    ) -> None:
        if strategy not in ("low", "high", "mid"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.k = k
        self.strategy = strategy
        self.text_key = text_key
        self.scorer = scorer or PerplexityScorer(lang=lang, model_name=model_name)

    def select(self, samples: list[dict]) -> list[dict]:
        if self.k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        if self.text_key not in df.columns:
            df[self.text_key] = [extract_text(s) for s in samples]

        ppl_scores = self.scorer.eval(df, input_key=self.text_key)
        ordered: list[tuple[float, int]] = []
        for i, s in enumerate(samples):
            ordered.append((float(ppl_scores[i]), i))

        if self.strategy == "low":
            ordered.sort(key=lambda x: x[0])
        elif self.strategy == "high":
            ordered.sort(key=lambda x: x[0], reverse=True)
        else:
            mean_ppl = sum(v[0] for v in ordered) / len(ordered)
            ordered.sort(key=lambda x: abs(math.log(x[0]) - math.log(mean_ppl)))

        return [
            {
                **samples[idx],
                "meta": {
                    "selector": "PerplexityBasedSelector",
                    "ppl": ppl,
                    "strategy": self.strategy,
                },
            }
            for ppl, idx in ordered[: min(self.k, len(ordered))]
        ]
