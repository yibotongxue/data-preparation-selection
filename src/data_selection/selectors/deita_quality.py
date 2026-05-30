from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)

from data_selection.config import MaybeConfig, maybe_create


class DeitaQualitySelector:
    """Select samples by Deita quality x complexity composite score."""

    def __init__(
        self,
        k: int = 100,
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        max_length: int = 512,
        quality_scorer: MaybeConfig[DeitaQualityScorer] = None,
        complexity_scorer: MaybeConfig[DeitaComplexityScorer] = None,
    ) -> None:
        self.k = k
        self.quality_scorer = maybe_create(quality_scorer) or DeitaQualityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )
        self.complexity_scorer = maybe_create(
            complexity_scorer
        ) or DeitaComplexityScorer(
            device=device, model_cache_dir=model_cache_dir, max_length=max_length
        )

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        df["instruction"] = [_msg_role(s, "user") for s in samples]
        df["output"] = [_msg_role(s, "assistant") for s in samples]

        q_scores = self.quality_scorer.eval(df)
        c_scores = self.complexity_scorer.eval(df)

        scored: list[tuple[float, float, float, Mapping[str, Any]]] = []
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
            for composite, q, c, s in scored[: min(self.k, len(scored))]
        ]


def _msg_role(sample: Mapping[str, Any], role: str) -> str:
    for msg in sample.get("messages", []):
        if msg.get("role") == role:
            return str(msg.get("content", ""))
    return ""
