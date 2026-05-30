from __future__ import annotations

from configs.random import selector as random_selector

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import CompositeSelector, LengthBasedSelector


def selector():
    random_cfg = random_selector()
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_composite.jsonl",
        "selector": CustomOmegaConfig.of(
            CompositeSelector,
            k=100000,
            selectors=[
                random_cfg["selector"],
                CustomOmegaConfig.of(LengthBasedSelector, k=100000),
            ],
        ),
    }
