from data_selection.selectors.embedding_similarity import (
    EmbeddingSimilaritySelector,
)


class TestEmbeddingSimilaritySelection:
    def test_select_basic(self):
        proxy = [1.0, 0.0, 0.0]
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 1.0, 0.0]},
            {"instruction": "c", "embedding": [0.5, 0.0, 0.0]},
        ]
        result = EmbeddingSimilaritySelector(k=2, domain_proxy=proxy).select(samples)
        assert result[0]["instruction"] == "a"
        assert result[0]["meta"]["similarity"] == 1.0

    def test_select_auto_proxy(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 1.0]},
            {"instruction": "c", "embedding": [1.0, 0.0]},
        ]
        result = EmbeddingSimilaritySelector(k=2).select(samples)
        assert len(result) == 2

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "embedding": [1.0, 0.0]}]
        result = EmbeddingSimilaritySelector(k=0).select(samples)
        assert result == []

    def test_select_empty(self):
        result = EmbeddingSimilaritySelector(k=3).select([])
        assert result == []

    def test_select_no_embedding(self):
        result = EmbeddingSimilaritySelector(k=2, domain_proxy=[1.0, 0.0]).select(
            [{"instruction": "a"}, {"instruction": "b"}]
        )
        assert result == []

    def test_cosine_similarity_perfect(self):
        proxy = [2.0, 0.0]
        samples = [
            {"instruction": "a", "embedding": [2.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 2.0]},
        ]
        result = EmbeddingSimilaritySelector(k=1, domain_proxy=proxy).select(samples)
        assert result[0]["instruction"] == "a"

    def test_cosine_zero_vector(self):
        proxy = [0.0, 0.0]
        samples = [{"instruction": "a", "embedding": [1.0, 0.0]}]
        result = EmbeddingSimilaritySelector(k=1, domain_proxy=proxy).select(samples)
        assert len(result) == 1
