from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import LengthBasedSelector


def selector():
    return {
        "input": "data/input.jsonl",
        "output": "data/output_length.jsonl",
        "selector": CustomOmegaConfig.of(LengthBasedSelector, k=100),
    }
