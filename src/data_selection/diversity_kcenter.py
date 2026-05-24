import math
import random


class DiversityKCenterSelection:
    """TSDS algorithm: greedy k-center in embedding space for maximum diversity.

    Starting from a random seed, iteratively selects the sample farthest
    from all already-selected samples in embedding space.
    """

    def __init__(
        self, embedding_key: str = "embedding", seed: int | None = None
    ) -> None:
        self.embedding_key = embedding_key
        self.seed = seed

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []
        k = min(k, len(samples))

        valid = [s for s in samples if self.embedding_key in s]
        if not valid:
            return []
        k = min(k, len(valid))

        rng = random.Random(self.seed)
        selected_idx: list[int] = []
        min_dist: list[float] = [float("inf")] * len(valid)

        first = rng.randrange(len(valid))
        selected_idx.append(first)

        for _ in range(1, k):
            last_emb = valid[selected_idx[-1]][self.embedding_key]
            best_idx = -1
            best_dist = -1.0
            for i in range(len(valid)):
                if i in selected_idx:
                    continue
                d = _euclidean_distance(valid[i][self.embedding_key], last_emb)
                if d < min_dist[i]:
                    min_dist[i] = d
                if min_dist[i] > best_dist:
                    best_dist = min_dist[i]
                    best_idx = i
            if best_idx >= 0:
                selected_idx.append(best_idx)

        return [valid[i] for i in selected_idx]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
