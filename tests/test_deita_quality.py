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

        selector = DeitaQualitySelector(quality_scorer=mock_q, complexity_scorer=mock_c)
        result = selector.select(samples, k=2)
        assert len(result) == 2
        assert result[0]["instruction"] == "c"
        assert result[0]["meta"]["composite_score"] == 15.0
        assert result[0]["meta"]["quality_score"] == 3.0
        assert result[0]["meta"]["complexity_score"] == 5.0

    def test_select_k_zero(self):
        samples = [{"instruction": "a"}]
        result = DeitaQualitySelector(
            quality_scorer=MagicMock(), complexity_scorer=MagicMock()
        ).select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = DeitaQualitySelector(
            quality_scorer=MagicMock(), complexity_scorer=MagicMock()
        ).select([], k=3)
        assert result == []
