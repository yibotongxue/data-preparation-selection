from __future__ import annotations

from data_selection.selectors.random_selection import RandomSelector


class TestRandomSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": f"task {i}", "output": f"result {i}"} for i in range(10)
        ]
        result = RandomSelector(k=3, seed=42).select(samples)
        assert len(result) == 3
        for r in result:
            assert "meta" in r
            assert r["meta"]["selector"] == "RandomSelector"

    def test_select_deterministic(self):
        samples = [{"instruction": f"task {i}"} for i in range(10)]
        a = RandomSelector(k=3, seed=42).select(samples)
        b = RandomSelector(k=3, seed=42).select(samples)
        assert [s["instruction"] for s in a] == [s["instruction"] for s in b]

    def test_select_k_zero(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector(k=0).select(samples)
        assert result == []

    def test_select_k_larger_than_pool(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector(k=10).select(samples)
        assert len(result) == 5

    def test_select_empty_pool(self):
        result = RandomSelector(k=3).select([])
        assert result == []

    def test_select_negative_k(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector(k=-1).select(samples)
        assert result == []
