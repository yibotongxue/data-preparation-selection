from __future__ import annotations

from data_selection.selectors.length_based import LengthBasedSelector


class TestLengthBasedSelection:
    def test_select_by_instruction_output(self):
        samples = [
            {"instruction": "hi", "output": "ok"},
            {"instruction": "hello world " * 10, "output": "long response " * 10},
            {"instruction": "medium text " * 3, "output": "also medium " * 3},
        ]
        result = LengthBasedSelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0]["instruction"] == samples[1]["instruction"]
        assert result[0]["meta"]["length"] > result[1]["meta"]["length"]

    def test_select_k_zero(self):
        samples = [{"instruction": "hi", "output": "ok"}]
        result = LengthBasedSelector(k=0).select(samples)
        assert result == []

    def test_select_empty(self):
        result = LengthBasedSelector(k=3).select([])
        assert result == []

    def test_select_missing_fields(self):
        samples = [{"instruction": "hi"}, {"output": "ok"}, {}]
        result = LengthBasedSelector(k=3).select(samples)
        assert len(result) == 3
