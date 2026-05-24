from data_selection.selectors.random_selection import RandomSelector


class TestRandomSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": f"task {i}", "output": f"result {i}"} for i in range(10)
        ]
        selector = RandomSelector(seed=42)
        result = selector.select(samples, k=3)
        assert len(result) == 3
        for r in result:
            assert "meta" in r
            assert r["meta"]["selector"] == "RandomSelector"

    def test_select_deterministic(self):
        samples = [{"instruction": f"task {i}"} for i in range(10)]
        a = RandomSelector(seed=42).select(samples, k=3)
        b = RandomSelector(seed=42).select(samples, k=3)
        assert [s["instruction"] for s in a] == [s["instruction"] for s in b]

    def test_select_k_zero(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector().select(samples, k=0)
        assert result == []

    def test_select_k_larger_than_pool(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector().select(samples, k=10)
        assert len(result) == 5

    def test_select_empty_pool(self):
        result = RandomSelector().select([], k=3)
        assert result == []

    def test_select_negative_k(self):
        samples = [{"instruction": f"task {i}"} for i in range(5)]
        result = RandomSelector().select(samples, k=-1)
        assert result == []
