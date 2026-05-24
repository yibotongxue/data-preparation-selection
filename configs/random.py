from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import RandomSelector


def selector():
    return {
        "input": "data/input.jsonl",
        "output": "data/output_random.jsonl",
        "selector": CustomOmegaConfig.of(RandomSelector, k=100, seed=42),
    }
