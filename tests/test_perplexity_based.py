import pytest

from data_selection.perplexity_based import PerplexityBasedSelection


class TestPerplexityBasedSelection:
    def test_select_low_ppl(self):
        samples = [
            {"instruction": "a", "ppl": 10.0},
            {"instruction": "b", "ppl": 5.0},
            {"instruction": "c", "ppl": 20.0},
        ]
        result = PerplexityBasedSelection(strategy="low").select(samples, k=2)
        assert len(result) == 2
        assert result[0]["ppl"] == 5.0
        assert result[1]["ppl"] == 10.0

    def test_select_high_ppl(self):
        samples = [
            {"instruction": "a", "ppl": 10.0},
            {"instruction": "b", "ppl": 5.0},
            {"instruction": "c", "ppl": 20.0},
        ]
        result = PerplexityBasedSelection(strategy="high").select(samples, k=2)
        assert result[0]["ppl"] == 20.0
        assert result[1]["ppl"] == 10.0

    def test_select_mid_ppl(self):
        samples = [
            {"instruction": "a", "ppl": 10.0},
            {"instruction": "b", "ppl": 50.0},
            {"instruction": "c", "ppl": 20.0},
        ]
        result = PerplexityBasedSelection(strategy="mid").select(samples, k=2)
        assert len(result) == 2

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "ppl": 10.0}]
        result = PerplexityBasedSelection().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = PerplexityBasedSelection().select([], k=3)
        assert result == []

    def test_select_no_ppl_field(self):
        samples = [{"instruction": "a"}, {"instruction": "b"}]
        result = PerplexityBasedSelection().select(samples, k=2)
        assert result == []

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            PerplexityBasedSelection(strategy="invalid")
