from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import RandomSelector


def selector():
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_random.jsonl",
        "selector": CustomOmegaConfig.of(
            RandomSelector,
            k=100000,
            seed=42,
        ),
    }
