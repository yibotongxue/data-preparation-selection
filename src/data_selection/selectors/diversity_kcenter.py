import random

import numpy as np


class DiversityKCenterSelection:
    """TSDS algorithm (OpenDCAI/DataFlex): greedy k-center in embedding
    space for maximum diversity coverage.

    Starting from a random seed, iteratively selects the sample farthest
    from all already-selected samples in Euclidean embedding space.
    Uses numpy for vectorized distance computation.
    """

    def __init__(
        self, embedding_key: str = "embedding", seed: int | None = None
    ) -> None:
        self.embedding_key = embedding_key
        self.seed = seed

    def select(self, samples: list[dict], k: int) -> list[dict]:
        if k <= 0 or not samples:
            return []

        valid = [s for s in samples if self.embedding_key in s]
        if not valid:
            return []

        embeddings = np.array([s[self.embedding_key] for s in valid], dtype=np.float64)
        k = min(k, len(valid))

        rng = random.Random(self.seed)
        selected: list[int] = [rng.randrange(len(valid))]
        min_dists: np.ndarray = np.full(len(valid), np.inf)

        for _ in range(1, k):
            last_emb = embeddings[selected[-1]]
            dists = np.linalg.norm(embeddings - last_emb, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1.0
            best = int(np.argmax(min_dists))
            selected.append(best)

        return [
            {
                **valid[i],
                "meta": {
                    "selector": "DiversityKCenterSelection",
                    "min_distance": round(float(min_dists[i]), 6),
                },
            }
            for i in selected
        ]
