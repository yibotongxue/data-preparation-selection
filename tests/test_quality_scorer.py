from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from data_selection.selectors.quality_scorer import QualityScorerSelector


def _sample(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class TestQualityScorerSelection:
    def test_select_fineweb_edu(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock_edu = MagicMock()
        mock_edu.eval.return_value = np.array([0.8, 0.3, 0.9])
        result = QualityScorerSelector(
            k=2, strategy="fineweb_edu", edu_scorer=mock_edu
        ).select(samples)
        assert result[0]["meta"]["fineweb_edu_score"] == 0.9

    def test_select_pairqual(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock_pq = MagicMock()
        mock_pq.eval.return_value = np.array([0.2, 0.9, 0.5])
        result = QualityScorerSelector(
            k=2, strategy="pairqual", pq_scorer=mock_pq
        ).select(samples)
        assert result[0]["meta"]["pairqual_score"] == 0.9

    def test_select_composite(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock_edu = MagicMock()
        mock_edu.eval.return_value = np.array([0.8, 0.3, 0.9])
        mock_pq = MagicMock()
        mock_pq.eval.return_value = np.array([0.2, 0.9, 0.5])
        result = QualityScorerSelector(
            k=2, strategy="composite", edu_scorer=mock_edu, pq_scorer=mock_pq
        ).select(samples)
        assert result[0]["meta"]["strategy"] == "composite"

    def test_select_k_zero(self):
        result = QualityScorerSelector(
            k=0, edu_scorer=MagicMock(), pq_scorer=MagicMock()
        ).select([_sample("hi")])
        assert result == []

    def test_select_empty(self):
        result = QualityScorerSelector(
            k=3, edu_scorer=MagicMock(), pq_scorer=MagicMock()
        ).select([])
        assert result == []

    def test_invalid_strategy(self):
        import pytest

        with pytest.raises(ValueError):
            QualityScorerSelector(strategy="invalid")
