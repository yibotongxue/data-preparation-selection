from __future__ import annotations

from data_selection.selectors.random_selection import RandomSelector


def _sample(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


class TestRandomSelection:
    def test_select_basic(self):
        samples = [_sample(f"task {i}") for i in range(10)]
        result = RandomSelector(k=3, seed=42).select(samples)
        assert len(result) == 3
        for r in result:
            assert "meta" in r
            assert r["meta"]["selector"] == "RandomSelector"

    def test_select_deterministic(self):
        samples = [_sample(f"task {i}") for i in range(10)]
        a = RandomSelector(k=3, seed=42).select(samples)
        b = RandomSelector(k=3, seed=42).select(samples)
        assert [s["messages"] for s in a] == [s["messages"] for s in b]

    def test_select_k_zero(self):
        result = RandomSelector(k=0).select([_sample("x")])
        assert result == []

    def test_select_k_larger_than_pool(self):
        samples = [_sample(f"task {i}") for i in range(5)]
        result = RandomSelector(k=10).select(samples)
        assert len(result) == 5

    def test_select_empty_pool(self):
        assert RandomSelector(k=3).select([]) == []

    def test_select_negative_k(self):
        result = RandomSelector(k=-1).select([_sample("x")])
        assert result == []
