from data_selection.selectors.length_based import LengthBasedSelection


class TestLengthBasedSelection:
    def test_select_by_instruction_output(self):
        samples = [
            {"instruction": "hi", "output": "ok"},
            {"instruction": "hello world " * 10, "output": "long response " * 10},
            {"instruction": "medium text " * 3, "output": "also medium " * 3},
        ]
        result = LengthBasedSelection().select(samples, k=2)
        assert len(result) == 2
        assert result[0]["instruction"] == samples[1]["instruction"]
        assert "meta" in result[0]
        assert result[0]["meta"]["length"] > result[1]["meta"]["length"]

    def test_select_by_conversations(self):
        samples = [
            {"conversations": [{"messages": [{"content": "short"}]}]},
            {"conversations": [{"messages": [{"content": "a" * 100}]}]},
            {"conversations": [{"messages": [{"content": "a" * 50}]}]},
        ]
        result = LengthBasedSelection().select(samples, k=2)
        assert result[0]["meta"]["length"] >= result[1]["meta"]["length"]

    def test_select_k_zero(self):
        samples = [{"instruction": "hi", "output": "ok"}]
        result = LengthBasedSelection().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = LengthBasedSelection().select([], k=3)
        assert result == []

    def test_select_missing_fields(self):
        samples = [
            {"instruction": "hi"},
            {"output": "ok"},
            {},
        ]
        result = LengthBasedSelection().select(samples, k=3)
        assert len(result) == 3
