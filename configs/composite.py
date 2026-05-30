from __future__ import annotations

from configs.random import selector as random_selector

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import CompositeSelector, LengthBasedSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="data/input.jsonl",
        output="data/output_composite.jsonl",
    )


def selector():
    random_cfg = random_selector()
    return {
        "selector": CustomOmegaConfig.of(
            CompositeSelector,
            k=50,
            selectors=[
                random_cfg["selector"],
                CustomOmegaConfig.of(LengthBasedSelector, k=50),
            ],
        ),
    }
