from typing import Any

import pandas as pd
from dataflow.operators.eval import FineWebEduScorer, PairQualScorer

from data_selection.utils import extract_text


class QualityScorerSelection:
    """Select samples by educational/quality score using DataFlow scorers.

    Supports three scoring strategies:
      - "fineweb_edu": FineWebEduScorer (HuggingFaceTB/fineweb-edu-classifier)
      - "pairqual": PairQualScorer (BGE-based, supports en/zh)
      - "composite": average of both scores
    """

    def __init__(
        self,
        strategy: str = "composite",
        text_key: str = "text",
        device: str = "cuda",
        model_cache_dir: str = "./dataflow_cache",
        lang: str = "en",
        edu_scorer: Any = None,
        pq_scorer: Any = None,
    ) -> None:
        if strategy not in ("fineweb_edu", "pairqual", "composite"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        self.text_key = text_key

        if strategy in ("fineweb_edu", "composite"):
            self.edu_scorer = edu_scorer or FineWebEduScorer(
                model_cache_dir=model_cache_dir, device=device
            )
        else:
            self.edu_scorer = None

        if strategy in ("pairqual", "composite"):
            self.pq_scorer = pq_scorer or PairQualScorer(
                model_cache_dir=model_cache_dir, device=device, lang=lang
            )
        else:
            self.pq_scorer = None

    def select(self, samples: list[dict], k: int) -> list[dict]:
        if k <= 0 or not samples:
            return []

        df = pd.DataFrame(samples)
        if self.text_key not in df.columns:
            df[self.text_key] = [extract_text(s) for s in samples]

        edu_scores: list[float] = []
        pq_scores: list[float] = []

        if self.edu_scorer is not None:
            edu_scores = self.edu_scorer.eval(df, input_key=self.text_key).tolist()
        if self.pq_scorer is not None:
            pq_scores = self.pq_scorer.eval(df, input_key=self.text_key).tolist()

        scored = []
        for i, s in enumerate(samples):
            e = float(edu_scores[i]) if edu_scores else None
            p = float(pq_scores[i]) if pq_scores else None
            if self.strategy == "fineweb_edu":
                score = e or 0.0
            elif self.strategy == "pairqual":
                score = p or 0.0
            else:
                score = ((e or 0.0) + (p or 0.0)) / 2.0
            scored.append((score, e, p, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                **s,
                "meta": {
                    "selector": "QualityScorerSelection",
                    "strategy": self.strategy,
                    "fineweb_edu_score": round(e, 6) if e is not None else None,
                    "pairqual_score": round(p, 6) if p is not None else None,
                },
            }
            for score, e, p, s in scored[: min(k, len(scored))]
        ]
