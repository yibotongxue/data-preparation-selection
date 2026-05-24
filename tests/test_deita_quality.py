from __future__ import annotations

from unittest.mock import MagicMock

from data_selection.selectors.deita_quality import DeitaQualitySelector


class TestDeitaQualitySelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "a", "output": "x"},
            {"instruction": "b", "output": "y"},
            {"instruction": "c", "output": "z"},
        ]
        mock_q = MagicMock()
        mock_q.eval.return_value = [4.0, 2.0, 3.0]
        mock_c = MagicMock()
        mock_c.eval.return_value = [3.0, 1.0, 5.0]

        result = DeitaQualitySelector(
            k=2, quality_scorer=mock_q, complexity_scorer=mock_c
        ).select(samples)
        assert len(result) == 2
        assert result[0]["instruction"] == "c"
        assert result[0]["meta"]["composite_score"] == 15.0

    def test_select_k_zero(self):
        result = DeitaQualitySelector(
            k=0, quality_scorer=MagicMock(), complexity_scorer=MagicMock()
        ).select([{"instruction": "a"}])
        assert result == []

    def test_select_empty(self):
        result = DeitaQualitySelector(
            k=3, quality_scorer=MagicMock(), complexity_scorer=MagicMock()
        ).select([])
        assert result == []
