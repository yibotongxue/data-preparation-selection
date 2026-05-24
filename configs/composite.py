from __future__ import annotations

from configs.random import selector as random_selector

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import CompositeSelector, LengthBasedSelector


def selector():
    random_cfg = random_selector()
    return {
        "input": "data/input.jsonl",
        "output": "data/output_composite.jsonl",
        "selector": CustomOmegaConfig.of(
            CompositeSelector,
            k=50,
            selectors=[
                random_cfg["selector"],
                CustomOmegaConfig.of(LengthBasedSelector, k=50),
            ],
        ),
    }
