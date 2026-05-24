import math
from typing import Any

import pandas as pd
from dataflow.operators.eval import PerplexityScorer

from data_selection.utils import extract_text


class PerplexityBasedSelection:
    """Select samples by perplexity using DataFlow's Kenlm-based scorer.

    Supports three strategies:
      - "low": lowest PPL (most fluent)
      - "high": highest PPL (most difficult)
      - "mid": PPL closest to mean (sweet spot)

    Requires Kenlm model files downloaded from HuggingFace.
    """

    def __init__(
        self,
        strategy: str = "low",
        text_key: str = "text",
        lang: str = "en",
        model_name: str = "dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
        scorer: Any = None,
    ) -> None:
        if strategy not in ("low", "high", "mid"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.text_key = text_key
        self.scorer = scorer or PerplexityScorer(lang=lang, model_name=model_name)

    def select(self, samples: list[dict], k: int) -> list[dict]:
        if k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        if self.text_key not in df.columns:
            df[self.text_key] = [extract_text(s) for s in samples]

        ppl_scores = self.scorer.eval(df, input_key=self.text_key)
        valid = [(float(ppl_scores[i]), s) for i, s in enumerate(samples)]

        if self.strategy == "low":
            valid.sort(key=lambda x: x[0])
        elif self.strategy == "high":
            valid.sort(key=lambda x: x[0], reverse=True)
        else:
            mean_ppl = sum(v[0] for v in valid) / len(valid)
            valid.sort(key=lambda x: abs(math.log(x[0]) - math.log(mean_ppl)))

        return [s for _, s in valid[: min(k, len(valid))]]
