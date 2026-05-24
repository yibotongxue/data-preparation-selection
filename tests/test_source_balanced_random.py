from data_selection.selectors.source_balanced_random import (
    SourceBalancedRandomSelector,
)


class TestSourceBalancedRandomSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": f"s{i}", "source": src}
            for i, src in enumerate(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        ]
        selector = SourceBalancedRandomSelector(seed=42)
        result = selector.select(samples, k=6)
        assert len(result) == 6

    def test_each_source_represented(self):
        samples = [
            {"instruction": f"s{i}", "source": src}
            for i, src in enumerate(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
        ]
        selector = SourceBalancedRandomSelector(seed=42)
        result = selector.select(samples, k=30)
        sources = {s["source"] for s in result}
        assert sources == {"a", "b", "c"}

    def test_select_k_zero(self):
        samples = [{"instruction": "x", "source": "a"}]
        result = SourceBalancedRandomSelector().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = SourceBalancedRandomSelector().select([], k=5)
        assert result == []

    def test_missing_source_key(self):
        samples = [{"instruction": "x"}, {"instruction": "y", "source": "a"}]
        selector = SourceBalancedRandomSelector(seed=42)
        result = selector.select(samples, k=2)
        assert len(result) == 2

    def test_custom_source_key(self):
        samples = [
            {"instruction": "x", "dataset": "a"},
            {"instruction": "y", "dataset": "b"},
            {"instruction": "z", "dataset": "a"},
        ]
        selector = SourceBalancedRandomSelector(source_key="dataset", seed=42)
        result = selector.select(samples, k=2)
        assert len(result) == 2
