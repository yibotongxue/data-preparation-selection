from __future__ import annotations

from dataflow.operators.eval import DeitaComplexityScorer, DeitaQualityScorer

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import DeitaQualitySelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="data/input.jsonl",
        output="data/output_deita.jsonl",
    )


def selector():
    return {
        "selector": CustomOmegaConfig.of(
            DeitaQualitySelector,
            k=100,
            device="cpu",
            quality_scorer=CustomOmegaConfig.of(DeitaQualityScorer, device="cpu"),
            complexity_scorer=CustomOmegaConfig.of(DeitaComplexityScorer, device="cpu"),
        ),
    }
