from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from dataflex.offline_selector.offline_near_selector import offline_near_Selector

from data_selection.utils import extract_text


class EmbeddingSimilaritySelector:
    """Near algorithm (DataFlex): select samples most similar to domain proxy.

    Uses DataFlex's offline_near_Selector with FAISS IVFFlat index.
    Text is extracted from instruction/output or conversations via
    extract_text(), then written as Alpaca JSON for DataFlex.
    """

    def __init__(
        self,
        k: int = 100,
        domain_proxy_text: str | None = None,
        embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embed_method: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.k = k
        self.domain_proxy_text = domain_proxy_text
        self.embed_model = embed_model
        self.embed_method = embed_method
        self.batch_size = batch_size

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.k <= 0 or not samples:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidates.json")
            indices_path = os.path.join(tmpdir, "top_indices.npy")

            _write_alpaca_json(samples, candidate_path)

            if self.domain_proxy_text:
                query_path = os.path.join(tmpdir, "query.json")
                _write_alpaca_json(
                    [{"instruction": self.domain_proxy_text, "output": ""}],
                    query_path,
                )
            else:
                query_path = candidate_path

            near = offline_near_Selector(
                candidate_path=candidate_path,
                query_path=query_path,
                embed_model=self.embed_model,
                embed_method=self.embed_method,
                batch_size=self.batch_size,
                save_indices_path=indices_path,
                max_K=max(self.k, 64),
            )
            near.selector()

            top_indices: np.ndarray = np.load(indices_path)
            selected_indices = top_indices[0][: min(self.k, top_indices.shape[1])]

        result: list[dict[str, Any]] = []
        for i in selected_indices:
            idx = int(i)
            result.append(
                {
                    **samples[idx],
                    "meta": {
                        "selector": "EmbeddingSimilaritySelector",
                        "neighbor_rank": int(list(selected_indices).index(i)),
                    },
                }
            )
        return result


def _write_alpaca_json(
    samples: Sequence[Mapping[str, Any]], path: str
) -> None:
    items: list[dict[str, str]] = []
    for s in samples:
        text = extract_text(s)
        items.append({"instruction": text, "input": "", "output": ""})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
