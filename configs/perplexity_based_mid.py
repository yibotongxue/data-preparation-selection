from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import PerplexityBasedSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output="data/output_perplexity_mid.jsonl",
    )


def selector():
    return {
        "selector": CustomOmegaConfig.of(
            PerplexityBasedSelector,
            k=100000,
            strategy="mid",
            text_key="text",
            lang="en",
            model_name="dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
        ),
    }
