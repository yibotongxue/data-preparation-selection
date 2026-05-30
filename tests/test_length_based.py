from __future__ import annotations

from data_selection.selectors.length_based import LengthBasedSelector


class TestLengthBasedSelection:
    def test_select_basic(self):
        samples = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "hello world " * 10}]},
            {"messages": [{"role": "user", "content": "medium"}]},
        ]
        result = LengthBasedSelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0]["messages"] == samples[1]["messages"]
        assert result[0]["meta"]["length"] > result[1]["meta"]["length"]

    def test_select_k_zero(self):
        result = LengthBasedSelector(k=0).select(
            [{"messages": [{"role": "user", "content": "hi"}]}]
        )
        assert result == []

    def test_select_empty(self):
        assert LengthBasedSelector(k=3).select([]) == []
