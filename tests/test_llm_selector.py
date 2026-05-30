from __future__ import annotations

from unittest.mock import MagicMock

from data_selection.selectors.llm_selector import LLMAsSelector


def _sample(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class TestLLMAsSelector:
    def test_select_with_scorer(self):
        samples = [_sample("short"), _sample("a" * 100), _sample("medium")]
        mock = MagicMock()
        mock.eval.return_value = [[3.0] * 6, [5.0] * 6, [1.0] * 6]
        result = LLMAsSelector(k=2, scorer=mock).select(samples)
        assert result[0]["meta"]["average_score"] == 5.0

    def test_select_no_scorer_fallback(self):
        samples = [
            _sample("hi"),
            _sample("hello world " * 10),
            _sample("ok"),
        ]
        result = LLMAsSelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0]["meta"]["dimension_scores"] is None

    def test_select_k_zero(self):
        result = LLMAsSelector(k=0).select([_sample("hi")])
        assert result == []

    def test_select_empty(self):
        assert LLMAsSelector(k=3).select([]) == []
