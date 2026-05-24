import numpy as np


class EmbeddingSimilaritySelection:
    """Near algorithm (OpenDCAI/DataFlex): select samples most similar to
    a domain proxy embedding via cosine similarity.

    Uses numpy for vectorized cosine similarity computation.
    """

    def __init__(self, embedding_key: str = "embedding") -> None:
        self.embedding_key = embedding_key

    def select(
        self,
        samples: list[dict],
        k: int,
        domain_proxy: list[float] | None = None,
        **kwargs,
    ) -> list[dict]:
        if k <= 0 or not samples:
            return []

        proxy = domain_proxy or kwargs.get("domain_proxy")
        valid = [s for s in samples if self.embedding_key in s]
        if not valid:
            return []

        if proxy is None:
            emb_matrix = np.array([s[self.embedding_key] for s in valid])
            proxy = emb_matrix.mean(axis=0).tolist()

        proxy_arr = np.array(proxy, dtype=np.float64)
        scores = []
        for s in valid:
            emb = np.array(s[self.embedding_key], dtype=np.float64)
            sim = _cosine_similarity(emb, proxy_arr)
            scores.append((sim, s))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scores[: min(k, len(scores))]]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
