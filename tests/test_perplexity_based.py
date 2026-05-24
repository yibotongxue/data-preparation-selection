from unittest.mock import MagicMock

import pytest

from data_selection.perplexity_based import PerplexityBasedSelection


class TestPerplexityBasedSelection:
    def test_select_low_ppl(self):
        samples = [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
        ]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]

        result = PerplexityBasedSelection(strategy="low", scorer=mock).select(
            samples, k=2
        )
        assert len(result) == 2
        assert result[0] == samples[1]
        assert result[1] == samples[0]

    def test_select_high_ppl(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]

        result = PerplexityBasedSelection(strategy="high", scorer=mock).select(
            samples, k=2
        )
        assert result[0] == samples[2]
        assert result[1] == samples[0]

    def test_select_mid_ppl(self):
        samples = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 50.0, 20.0]

        result = PerplexityBasedSelection(strategy="mid", scorer=mock).select(
            samples, k=2
        )
        assert len(result) == 2

    def test_select_k_zero(self):
        mock = MagicMock()
        result = PerplexityBasedSelection(scorer=mock).select([{"text": "a"}], k=0)
        assert result == []

    def test_select_empty(self):
        mock = MagicMock()
        result = PerplexityBasedSelection(scorer=mock).select([], k=3)
        assert result == []

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            PerplexityBasedSelection(strategy="invalid")
