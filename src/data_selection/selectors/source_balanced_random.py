import random
from collections import defaultdict


class SourceBalancedRandomSelection:
    """Sample proportionally from each source dataset to avoid domination."""

    def __init__(self, source_key: str = "source", seed: int | None = None) -> None:
        self.source_key = source_key
        self.seed = seed

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []
        rng = random.Random(self.seed)
        by_source: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_source[s.get(self.source_key, "__unknown__")].append(s)

        n_sources = len(by_source)
        per_source = k // n_sources
        remainder = k % n_sources

        selected: list[dict] = []
        sources = list(by_source.keys())
        rng.shuffle(sources)
        for i, src in enumerate(sources):
            pool = by_source[src]
            take = per_source + (1 if i < remainder else 0)
            take = min(take, len(pool))
            if take > 0:
                selected.extend(rng.sample(pool, take))

        return selected
