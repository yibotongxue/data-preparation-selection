from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import SourceBalancedRandomSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output="data/output_source_balanced.jsonl",
    )

def selector():
    return {
        "selector": CustomOmegaConfig.of(
            SourceBalancedRandomSelector,
            k=100000,
            source_key="source",
            seed=42,
        ),
    }
