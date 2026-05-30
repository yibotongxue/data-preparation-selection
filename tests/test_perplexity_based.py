from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_selection.selectors.perplexity_based import PerplexityBasedSelector


def _sample(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class TestPerplexityBasedSelection:
    def test_select_low_ppl(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]
        result = PerplexityBasedSelector(k=2, strategy="low", scorer=mock).select(
            samples
        )
        assert result[0]["meta"]["ppl"] == 5.0

    def test_select_high_ppl(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 5.0, 20.0]
        result = PerplexityBasedSelector(k=2, strategy="high", scorer=mock).select(
            samples
        )
        assert result[0]["meta"]["ppl"] == 20.0

    def test_select_mid_ppl(self):
        samples = [_sample("a"), _sample("b"), _sample("c")]
        mock = MagicMock()
        mock.eval.return_value = [10.0, 50.0, 20.0]
        result = PerplexityBasedSelector(k=2, strategy="mid", scorer=mock).select(
            samples
        )
        assert result[0]["meta"]["strategy"] == "mid"

    def test_select_k_zero(self):
        mock = MagicMock()
        result = PerplexityBasedSelector(k=0, scorer=mock).select([_sample("a")])
        assert result == []

    def test_select_empty(self):
        mock = MagicMock()
        result = PerplexityBasedSelector(k=3, scorer=mock).select([])
        assert result == []

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            PerplexityBasedSelector(strategy="invalid")
