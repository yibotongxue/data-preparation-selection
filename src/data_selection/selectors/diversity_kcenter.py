from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from dataflex.offline_selector.offline_tsds_selector import offline_tsds_Selector


class DiversityKCenterSelector:
    """TSDS algorithm (DataFlex): greedy k-center for maximum diversity.

    Uses DataFlex's offline_tsds_Selector with KDE density and FAISS.
    Computes sentence embeddings from instruction/output/conversations.
    """

    def __init__(
        self,
        k: int = 100,
        seed: int | None = None,
        embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embed_method: str = "auto",
        batch_size: int = 32,
        sigma: float = 0.75,
        alpha: float = 0.6,
    ) -> None:
        self.k = k
        self.seed = seed
        self.embed_model = embed_model
        self.embed_method = embed_method
        self.batch_size = batch_size
        self.sigma = sigma
        self.alpha = alpha

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidates.json")
            probs_path = os.path.join(tmpdir, "tsds_probs.npy")

            _write_alpaca_json(samples, candidate_path)

            tsds = offline_tsds_Selector(
                candidate_path=candidate_path,
                query_path=candidate_path,
                embed_model=self.embed_model,
                embed_method=self.embed_method,
                batch_size=self.batch_size,
                save_probs_path=probs_path,
                max_K=min(5000, max(len(samples), self.k * 2)),
                kde_K=min(1000, len(samples)),
                sigma=self.sigma,
                alpha=self.alpha,
            )
            probs: np.ndarray = tsds.selector()

        top_indices = np.argsort(-probs)[: min(self.k, len(probs))]

        result: list[dict[str, Any]] = []
        for i in top_indices:
            idx = int(i)
            result.append(
                {
                    **samples[idx],
                    "meta": {
                        "selector": "DiversityKCenterSelector",
                        "tsds_probability": round(float(probs[idx]), 6),
                    },
                }
            )
        return result


def _write_alpaca_json(samples: Sequence[Mapping[str, Any]], path: str) -> None:
    items: list[dict[str, str]] = []
    for s in samples:
        items.append(
            {
                "instruction": str(s.get("instruction", "")),
                "input": "",
                "output": str(s.get("output", "")),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
