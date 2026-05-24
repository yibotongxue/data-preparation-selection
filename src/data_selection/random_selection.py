import random


class RandomSelection:
    """Random baseline: uniformly sample k items from the candidate pool."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []
        rng = random.Random(self.seed)
        return rng.sample(samples, min(k, len(samples)))
