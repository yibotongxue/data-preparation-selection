from typing import Any

import pandas as pd
from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)


class DeitaQualitySelection:
    """Select samples by Deita quality x complexity composite score.

    Uses DataFlow's DeitaQualityScorer and DeitaComplexityScorer
    (Llama-based models from hkust-nlp) to score each sample, then
    selects top-k by quality * complexity.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        max_length: int = 512,
        instruction_key: str = "instruction",
        output_key: str = "output",
        quality_scorer: Any = None,
        complexity_scorer: Any = None,
    ) -> None:
        self.instruction_key = instruction_key
        self.output_key = output_key
        self.quality_scorer = quality_scorer or DeitaQualityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )
        self.complexity_scorer = complexity_scorer or DeitaComplexityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )

    def select(self, samples: list[dict], k: int) -> list[dict]:
        if k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        q_scores = self.quality_scorer.eval(
            df,
            input_instruction_key=self.instruction_key,
            input_output_key=self.output_key,
        )
        c_scores = self.complexity_scorer.eval(
            df,
            input_instruction_key=self.instruction_key,
            input_output_key=self.output_key,
        )

        scored = []
        for i, s in enumerate(samples):
            composite = float(q_scores[i]) * float(c_scores[i])
            scored.append((composite, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: min(k, len(scored))]]
