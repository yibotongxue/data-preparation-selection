from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import DiversityKCenterSelector


def selector():
    return {
        "input": "data/input.jsonl",
        "output": "data/output_diversity.jsonl",
        "selector": CustomOmegaConfig.of(
            DiversityKCenterSelector,
            k=100,
            seed=42,
        ),
    }
