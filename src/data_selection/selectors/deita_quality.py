from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)

from data_selection.config import MaybeConfig, maybe_create
from data_selection.score_cache import ScoreCache


class DeitaQualitySelector:
    """Select samples by Deita quality x complexity composite score."""

    def __init__(
        self,
        k: int = 100,
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        max_length: int = 512,
        instruction_key: str = "instruction",
        output_key: str = "output",
        quality_scorer: MaybeConfig[DeitaQualityScorer] = None,
        complexity_scorer: MaybeConfig[DeitaComplexityScorer] = None,
        scores_cache_path: str | None = None,
        batch_size: int = 256,
    ) -> None:
        self.k = k
        self.instruction_key = instruction_key
        self.output_key = output_key
        self.batch_size = batch_size
        self.scores_cache_path = scores_cache_path
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

        # Load or create cache
        cache = ScoreCache(self.scores_cache_path) if self.scores_cache_path else None
        missing = cache.missing_indices(len(samples)) if cache else list(range(len(samples)))

        if missing:
            print(f"[DeitaQualitySelector] Scoring {len(missing)} samples (cached: {len(samples) - len(missing)})")
            # Score in batches to allow incremental saving
            for batch_start in range(0, len(missing), self.batch_size):
                batch_indices = missing[batch_start : batch_start + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]
                df = pd.DataFrame(batch_samples)

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

                if cache:
                    entries = []
                    for j, idx in enumerate(batch_indices):
                        entries.append((idx, {
                            "quality": float(q_scores[j]),
                            "complexity": float(c_scores[j]),
                        }))
                    cache.put_batch(entries)
                else:
                    # No cache — store in a temporary dict
                    if not hasattr(self, "_temp_scores"):
                        self._temp_scores: dict[int, dict[str, float]] = {}
                    for j, idx in enumerate(batch_indices):
                        self._temp_scores[idx] = {
                            "quality": float(q_scores[j]),
                            "complexity": float(c_scores[j]),
                        }

        # Collect all scores
        all_scores = cache.all_scores() if cache else getattr(self, "_temp_scores", {})

        scored: list[tuple[float, float, float, Mapping[str, Any]]] = []
        for i, s in enumerate(samples):
            sc = all_scores.get(i)
            if sc is None:
                continue
            q = sc["quality"]
            c = sc["complexity"]
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
