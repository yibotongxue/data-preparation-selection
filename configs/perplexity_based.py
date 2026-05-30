from __future__ import annotations

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import PerplexityBasedSelector


def selector():
    return {
        "input": "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl",
        "output": "data/output_perplexity.jsonl",
        "selector": CustomOmegaConfig.of(
            PerplexityBasedSelector,
            k=100000,
            strategy="low",
            text_key="text",
            lang="en",
            model_name="dataflow/operators/eval/GeneralText/models/Kenlm/wikipedia",
        ),
    }
