from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import EmbeddingSimilaritySelector


def selector():
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_embedding_similarity.jsonl",
        "selector": CustomOmegaConfig.of(
            EmbeddingSimilaritySelector,
            k=100000,
            domain_proxy_text="Solve the following math problem step by step.",
            embed_model="/jizhicfs/linyibo/models/Qwen/Qwen3-Embedding-0.6B",
            embed_method="auto",
            batch_size=32,
        ),
    }
