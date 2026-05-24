class DeitaQualitySelection:
    """Select samples with the highest quality × complexity composite score.

    DEITA (Data-Efficient Instruction Tuning for Alignment) combines quality
    and complexity scores into a single metric.
    """

    def __init__(
        self,
        quality_key: str = "quality_score",
        complexity_key: str = "complexity_score",
    ) -> None:
        self.quality_key = quality_key
        self.complexity_key = complexity_key

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        def _composite(s: dict) -> float:
            q = float(s.get(self.quality_key, 0))
            c = float(s.get(self.complexity_key, 0))
            return q * c

        scored = sorted(samples, key=_composite, reverse=True)
        return scored[: min(k, len(scored))]
