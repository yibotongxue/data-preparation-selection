from typing import Any

import pandas as pd
from dataflow.operators.eval import MetaScorer

from data_selection.utils import extract_text


class LLMAsSelector:
    """Use an LLM to score and rank samples via DataFlow's MetaScorer.

    MetaScorer evaluates text across 6 dimensions (Text Structure,
    Diversity, Fluency, Safety, Educational Value, Content Accuracy)
    using an LLM backend via DataFlow's LLMServingABC interface.

    Requires an llm_serving object implementing LLMServingABC, or
    inject a mock scorer for testing.
    """

    def __init__(
        self,
        llm_serving: Any = None,
        dimensions: Any = None,
        scorer: Any = None,
    ) -> None:
        if scorer is not None:
            self.scorer = scorer
        elif llm_serving is not None:
            self.scorer = MetaScorer(
                llm_serving=llm_serving,
                dimensions=dimensions or [],
            )
        else:
            self.scorer = None

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        text_key = kwargs.get("text_key", "text")
        if text_key not in df.columns:
            df[text_key] = [extract_text(s) for s in samples]

        if self.scorer is not None:
            scores_2d = self.scorer.eval(df, input_key=text_key)
            scores = [float(sum(row)) / len(row) for row in scores_2d]
        else:
            scores = [float(len(extract_text(s).split())) for s in samples]

        paired = list(zip(scores, samples, strict=True))
        paired.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in paired[: min(k, len(paired))]]
