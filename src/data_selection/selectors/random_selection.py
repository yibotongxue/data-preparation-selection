import random


class RandomSelector:
    """Random baseline: uniformly sample k items from the candidate pool."""

    def __init__(self, k: int = 100, seed: int | None = None) -> None:
        self.k = k
        self.seed = seed

    def select(self, samples: list[dict]) -> list[dict]:
        if self.k <= 0 or not samples:
            return []
        rng = random.Random(self.seed)
        chosen = rng.sample(samples, min(self.k, len(samples)))
        return [
            {**s, "meta": {"selector": "RandomSelector", "seed": self.seed}}
            for s in chosen
        ]
