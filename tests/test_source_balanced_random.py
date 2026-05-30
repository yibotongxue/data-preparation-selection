from __future__ import annotations

from data_selection.selectors.source_balanced_random import (
    SourceBalancedRandomSelector,
)


def _sample(text: str, source: str) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "source": source,
    }


class TestSourceBalancedRandomSelection:
    def test_select_basic(self):
        samples = [
            _sample(f"s{i}", src)
            for i, src in enumerate(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        ]
        result = SourceBalancedRandomSelector(k=6, seed=42).select(samples)
        assert len(result) == 6

    def test_each_source_represented(self):
        samples = [
            _sample(f"s{i}", src)
            for i, src in enumerate(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
        ]
        result = SourceBalancedRandomSelector(k=30, seed=42).select(samples)
        sources = {s["source"] for s in result}
        assert sources == {"a", "b", "c"}

    def test_select_k_zero(self):
        result = SourceBalancedRandomSelector(k=0).select([_sample("x", "a")])
        assert result == []

    def test_select_empty(self):
        assert SourceBalancedRandomSelector(k=5).select([]) == []

    def test_missing_source_key(self):
        samples = [
            {"messages": [{"role": "user", "content": "x"}]},
            {**_sample("y", "a")},
        ]
        result = SourceBalancedRandomSelector(k=2, seed=42).select(samples)
        assert len(result) == 2

    def test_custom_source_key(self):
        samples = [
            {"messages": [{"role": "user", "content": "x"}], "dataset": "a"},
            {"messages": [{"role": "user", "content": "y"}], "dataset": "b"},
            {"messages": [{"role": "user", "content": "z"}], "dataset": "a"},
        ]
        result = SourceBalancedRandomSelector(
            k=2, source_key="dataset", seed=42
        ).select(samples)
        assert len(result) == 2
