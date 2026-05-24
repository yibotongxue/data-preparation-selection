import math

from data_selection.protocol import SelectionMethod


class EmbeddingSimilaritySelection(SelectionMethod):
    """Near algorithm: select samples most similar to a domain proxy embedding.

    Computes cosine similarity between each sample's embedding and a domain
    proxy embedding (e.g. mean of target domain examples).
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

        proxy = domain_proxy
        if proxy is None:
            proxy = kwargs.get("domain_proxy")
        if proxy is None:
            embeddings = [
                s[self.embedding_key] for s in samples if self.embedding_key in s
            ]
            if not embeddings:
                return []
            dim = len(embeddings[0])
            proxy = [
                sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)
            ]

        valid = [s for s in samples if self.embedding_key in s]
        if not valid:
            return []

        scored = []
        for s in valid:
            emb = s[self.embedding_key]
            sim = _cosine_similarity(emb, proxy)
            scored.append((sim, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: min(k, len(scored))]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
