from unittest.mock import MagicMock

import pytest

from data_selection.selectors.perplexity_based import PerplexityBasedSelector


class TestPerplexityBasedSelection:
    def test_select_low_ppl(self):
        samples = [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
        ]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]

        result = PerplexityBasedSelector(strategy="low", scorer=mock).select(
            samples, k=2
        )
        assert len(result) == 2
        assert result[0]["meta"]["ppl"] == 5.0
        assert result[1]["meta"]["ppl"] == 10.0

    def test_select_high_ppl(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]

        result = PerplexityBasedSelector(strategy="high", scorer=mock).select(
            samples, k=2
        )
        assert result[0]["meta"]["ppl"] == 20.0
        assert result[1]["meta"]["ppl"] == 10.0

    def test_select_mid_ppl(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 50.0, 20.0]

        result = PerplexityBasedSelector(strategy="mid", scorer=mock).select(
            samples, k=2
        )
        assert len(result) == 2
        assert result[0]["meta"]["strategy"] == "mid"

    def test_select_k_zero(self):
        mock = MagicMock()
        result = PerplexityBasedSelector(scorer=mock).select([{"text": "a"}], k=0)
        assert result == []

    def test_select_empty(self):
        mock = MagicMock()
        result = PerplexityBasedSelector(scorer=mock).select([], k=3)
        assert result == []

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            PerplexityBasedSelector(strategy="invalid")
