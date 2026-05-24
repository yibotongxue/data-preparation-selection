from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from data_selection.selectors.embedding_similarity import (
    EmbeddingSimilaritySelector,
)


class TestEmbeddingSimilaritySelection:
    def test_select_basic(self, monkeypatch):
        mock_near = MagicMock()
        monkeypatch.setattr(
            "data_selection.selectors.embedding_similarity.offline_near_Selector",
            mock_near,
        )
        indices = np.array([[1, 0, 2]])
        monkeypatch.setattr(
            "data_selection.selectors.embedding_similarity.np.load",
            lambda path: indices,
        )

        samples = [
            {"instruction": "a", "output": "x"},
            {"instruction": "b", "output": "y"},
            {"instruction": "c", "output": "z"},
        ]
        result = EmbeddingSimilaritySelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0] == {
            **samples[1],
            "meta": {"selector": "EmbeddingSimilaritySelector", "neighbor_rank": 0},
        }
        assert result[1] == {
            **samples[0],
            "meta": {"selector": "EmbeddingSimilaritySelector", "neighbor_rank": 1},
        }

    def test_select_k_zero(self):
        assert EmbeddingSimilaritySelector(k=0).select([{"instruction": "a"}]) == []

    def test_select_empty(self):
        assert EmbeddingSimilaritySelector(k=3).select([]) == []
