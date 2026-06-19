from __future__ import annotations

from dataflow.operators.eval import FineWebEduScorer, PairQualScorer

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import QualityScorerSelector
from data_selection.dataset import DatasetConfig


def dataset():
    return CustomOmegaConfig.of(
        DatasetConfig,
        path="/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        output="data/output_quality.jsonl",
    )

def selector():
    return {
        "selector": CustomOmegaConfig.of(
            QualityScorerSelector,
            k=100000,
            strategy="composite",
            text_key="text",
            device="cuda",
            model_cache_dir="./dataflow_cache",
            lang="en",
            edu_scorer=CustomOmegaConfig.of(
                FineWebEduScorer,
                model_cache_dir="./dataflow_cache",
                device="cuda",
            ),
            pq_scorer=CustomOmegaConfig.of(
                PairQualScorer,
                model_cache_dir="./dataflow_cache",
                device="cuda",
                lang="en",
                max_length=512,
            ),
        ),
    }
