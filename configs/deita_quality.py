from __future__ import annotations

from dataflow.operators.eval import DeitaComplexityScorer, DeitaQualityScorer

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import DeitaQualitySelector


def selector():
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_deita.jsonl",
        "selector": CustomOmegaConfig.of(
            DeitaQualitySelector,
            k=100000,
            device="cuda",
            model_cache_dir="./dataflow_cache",
            max_length=512,
            instruction_key="instruction",
            output_key="output",
            quality_scorer=CustomOmegaConfig.of(
                DeitaQualityScorer,
                device="cuda",
                model_cache_dir="./dataflow_cache",
                max_length=512,
            ),
            complexity_scorer=CustomOmegaConfig.of(
                DeitaComplexityScorer,
                device="cuda",
                model_cache_dir="./dataflow_cache",
                max_length=512,
            ),
        ),
    }
