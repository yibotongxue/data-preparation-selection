import math
import random

from data_selection.protocol import SelectionMethod


class DiversityKCenterSelection(SelectionMethod):
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
        selected_idx: set[int] = set()
        min_dist: list[float] = [float("inf")] * len(valid)

        first = rng.randrange(len(valid))
        selected_idx.add(first)

        for _ in range(1, k):
            last_emb = valid[first][self.embedding_key]
            for i in range(len(valid)):
                if i in selected_idx:
                    continue
                d = _euclidean_distance(valid[i][self.embedding_key], last_emb)
                if d < min_dist[i]:
                    min_dist[i] = d

            first = max(
                (i for i in range(len(valid)) if i not in selected_idx),
                key=lambda i: min_dist[i],
            )

        return [valid[i] for i in selected_idx]


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
