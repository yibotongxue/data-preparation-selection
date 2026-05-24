import random


class RandomSelection:
    """Random baseline: uniformly sample k items from the candidate pool."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def select(self, samples: list[dict], k: int) -> list[dict]:
        if k <= 0 or not samples:
            return []
        rng = random.Random(self.seed)
        chosen = rng.sample(samples, min(k, len(samples)))
        return [
            {**s, "meta": {"selector": "RandomSelection", "seed": self.seed}}
            for s in chosen
        ]
