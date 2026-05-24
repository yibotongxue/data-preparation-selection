import pandas as pd
from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)


class DeitaQualitySelector:
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
        quality_scorer: DeitaQualityScorer | None = None,
        complexity_scorer: DeitaComplexityScorer | None = None,
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
            q = float(q_scores[i])
            c = float(c_scores[i])
            composite = q * c
            scored.append((composite, q, c, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                **s,
                "meta": {
                    "selector": "DeitaQualitySelector",
                    "quality_score": round(q, 4),
                    "complexity_score": round(c, 4),
                    "composite_score": round(composite, 4),
                },
            }
            for composite, q, c, s in scored[: min(k, len(scored))]
        ]
