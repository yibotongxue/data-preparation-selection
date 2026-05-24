from __future__ import annotations

from dataflow.operators.eval import (
    DeitaComplexityScorer,
    DeitaQualityScorer,
)

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import DeitaQualitySelector


def selector():
    return {
        "input": "data/input.jsonl",
        "output": "data/output_deita.jsonl",
        "selector": CustomOmegaConfig.of(
            DeitaQualitySelector,
            k=100,
            device="cuda",
            quality_scorer=CustomOmegaConfig.of(DeitaQualityScorer, device="cuda"),
            complexity_scorer=CustomOmegaConfig.of(
                DeitaComplexityScorer, device="cuda"
            ),
        ),
    }
