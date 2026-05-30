from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import RandomSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="data/input.jsonl",
        output="data/output_random.jsonl",
    )


def selector():
    return {
        "selector": CustomOmegaConfig.of(RandomSelector, k=100, seed=42),
    }
