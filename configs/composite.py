from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import (
    CompositeSelector,
    LengthBasedSelector,
    RandomSelector,
)


def selector():
    return {
        "input": "data/input.jsonl",
        "output": "data/output_composite.jsonl",
        "selector": CustomOmegaConfig.of(
            CompositeSelector,
            k=50,
            selectors=[
                CustomOmegaConfig.of(RandomSelector, k=200, seed=42),
                CustomOmegaConfig.of(LengthBasedSelector, k=50),
            ],
        ),
    }
