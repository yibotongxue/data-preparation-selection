from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import EmbeddingSimilaritySelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output="data/output_embedding_similarity.jsonl",
    )

def selector():
    return {
        "selector": CustomOmegaConfig.of(
            EmbeddingSimilaritySelector,
            k=100000,
            domain_proxy_text="Solve the following math problem step by step.",
            embed_model="/jizhicfs/linyibo/models/Qwen/Qwen3-Embedding-0.6B",
            embed_method="auto",
            batch_size=64,
        ),
    }
