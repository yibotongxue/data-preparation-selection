from unittest.mock import MagicMock

import numpy as np

from data_selection.selectors.quality_scorer import QualityScorerSelection


class TestQualityScorerSelection:
    def test_select_fineweb_edu(self):
        samples = [
            {"instruction": "a", "output": "x"},
            {"instruction": "b", "output": "y"},
            {"instruction": "c", "output": "z"},
        ]
        mock_edu = MagicMock()
        mock_edu.eval.return_value = np.array([0.8, 0.3, 0.9])

        selector = QualityScorerSelection(strategy="fineweb_edu", edu_scorer=mock_edu)
        result = selector.select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[2]

    def test_select_pairqual(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock_pq = MagicMock()
        mock_pq.eval.return_value = np.array([0.2, 0.9, 0.5])

        selector = QualityScorerSelection(strategy="pairqual", pq_scorer=mock_pq)
        result = selector.select(samples, k=2)
        assert result[0] == samples[1]

    def test_select_composite(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock_edu = MagicMock()
        mock_edu.eval.return_value = np.array([0.8, 0.3, 0.9])
        mock_pq = MagicMock()
        mock_pq.eval.return_value = np.array([0.2, 0.9, 0.5])

        selector = QualityScorerSelection(
            strategy="composite", edu_scorer=mock_edu, pq_scorer=mock_pq
        )
        result = selector.select(samples, k=2)
        assert len(result) == 2

    def test_select_k_zero(self):
        result = QualityScorerSelection(
            edu_scorer=MagicMock(), pq_scorer=MagicMock()
        ).select([{"text": "hi"}], k=0)
        assert result == []

    def test_select_empty(self):
        result = QualityScorerSelection(
            edu_scorer=MagicMock(), pq_scorer=MagicMock()
        ).select([], k=3)
        assert result == []

    def test_invalid_strategy(self):
        import pytest

        with pytest.raises(ValueError):
            QualityScorerSelection(strategy="invalid")
