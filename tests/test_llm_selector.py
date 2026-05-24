from __future__ import annotations

from unittest.mock import MagicMock

from data_selection.selectors.llm_selector import LLMAsSelector


class TestLLMAsSelector:
    def test_select_with_scorer(self):
        samples = [{"text": "short"}, {"text": "a" * 100}, {"text": "medium"}]
        mock = MagicMock()
        mock.eval.return_value = [[3.0] * 6, [5.0] * 6, [1.0] * 6]
        result = LLMAsSelector(k=2, scorer=mock).select(samples)
        assert result[0]["meta"]["average_score"] == 5.0
        assert result[0]["meta"]["dimension_scores"] == [5.0] * 6

    def test_select_no_scorer_fallback(self):
        samples = [
            {"instruction": "q", "output": "hi"},
            {"instruction": "q", "output": "hello world " * 10},
            {"instruction": "q", "output": "ok"},
        ]
        result = LLMAsSelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0]["meta"]["dimension_scores"] is None

    def test_select_k_zero(self):
        result = LLMAsSelector(k=0).select([{"text": "hi"}])
        assert result == []

    def test_select_empty(self):
        result = LLMAsSelector(k=3).select([])
        assert result == []
