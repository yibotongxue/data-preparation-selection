from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import LengthBasedSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="data/input.jsonl",
        output="data/output_length.jsonl",
    )


def selector():
    return {
        "selector": CustomOmegaConfig.of(LengthBasedSelector, k=100),
    }
