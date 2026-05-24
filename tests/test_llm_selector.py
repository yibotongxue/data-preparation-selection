from unittest.mock import MagicMock

from data_selection.llm_selector import LLMAsSelector


class TestLLMAsSelector:
    def test_select_with_scorer(self):
        samples = [
            {"text": "short"},
            {"text": "a" * 100},
            {"text": "medium"},
        ]
        mock = MagicMock()
        mock.eval.return_value = [[3.0] * 6, [5.0] * 6, [1.0] * 6]

        result = LLMAsSelector(scorer=mock).select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[1]

    def test_select_no_scorer_fallback(self):
        samples = [
            {"instruction": "q", "output": "hi"},
            {"instruction": "q", "output": "hello world " * 10},
            {"instruction": "q", "output": "ok"},
        ]
        result = LLMAsSelector().select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[1]

    def test_select_k_zero(self):
        result = LLMAsSelector().select([{"text": "hi"}], k=0)
        assert result == []

    def test_select_empty(self):
        result = LLMAsSelector().select([], k=3)
        assert result == []
