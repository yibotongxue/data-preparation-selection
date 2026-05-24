import math

from data_selection.protocol import SelectionMethod


class PerplexityBasedSelection(SelectionMethod):
    """Select samples based on perplexity scores from a fixed language model.

    Supports three strategies:
      - "low": lowest PPL (most fluent, easiest)
      - "high": highest PPL (most difficult, potentially noisy)
      - "mid": medium PPL (sweet spot between fluency and difficulty)
    """

    def __init__(self, ppl_key: str = "ppl", strategy: str = "low") -> None:
        if strategy not in ("low", "high", "mid"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.ppl_key = ppl_key
        self.strategy = strategy

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        valid = [s for s in samples if self.ppl_key in s]
        if not valid:
            return []

        if self.strategy == "low":
            sorted_samples = sorted(valid, key=lambda s: s[self.ppl_key])
        elif self.strategy == "high":
            sorted_samples = sorted(valid, key=lambda s: s[self.ppl_key], reverse=True)
        else:  # "mid"
            mean_ppl = sum(s[self.ppl_key] for s in valid) / len(valid)
            sorted_samples = sorted(
                valid,
                key=lambda s: abs(math.log(s[self.ppl_key]) - math.log(mean_ppl)),
            )

        return sorted_samples[: min(k, len(sorted_samples))]
