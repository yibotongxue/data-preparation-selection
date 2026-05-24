import numpy as np


class EmbeddingSimilaritySelector:
    """Near algorithm (OpenDCAI/DataFlex): select samples most similar to
    a domain proxy embedding via cosine similarity.

    If domain_proxy is None, it is computed as the mean embedding
    of all candidates at select time.
    """

    def __init__(
        self,
        k: int = 100,
        embedding_key: str = "embedding",
        domain_proxy: list[float] | None = None,
    ) -> None:
        self.k = k
        self.embedding_key = embedding_key
        self.domain_proxy = domain_proxy

    def select(self, samples: list[dict]) -> list[dict]:
        if self.k <= 0 or not samples:
            return []

        proxy = self.domain_proxy
        valid: list[tuple[int, dict]] = [
            (i, s) for i, s in enumerate(samples) if self.embedding_key in s
        ]
        if not valid:
            return []

        valid_indices, valid_samples = zip(*valid)
        if proxy is None:
            emb_matrix = np.array([s[self.embedding_key] for s in valid_samples])
            proxy = emb_matrix.mean(axis=0).tolist()

        proxy_arr = np.array(proxy, dtype=np.float64)
        scored: list[tuple[float, int, dict]] = []
        for orig_i, s in zip(valid_indices, valid_samples):
            emb = np.array(s[self.embedding_key], dtype=np.float64)
            sim = _cosine_similarity(emb, proxy_arr)
            scored.append((sim, orig_i, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                **s,
                "meta": {
                    "selector": "EmbeddingSimilaritySelector",
                    "similarity": round(sim, 6),
                },
            }
            for sim, _, s in scored[: min(self.k, len(scored))]
        ]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
