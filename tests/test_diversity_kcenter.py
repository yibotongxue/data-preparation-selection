from data_selection.selectors.diversity_kcenter import DiversityKCenterSelector


class TestDiversityKCenterSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 1.0]},
            {"instruction": "c", "embedding": [1.0, 0.1]},
            {"instruction": "d", "embedding": [0.0, 0.9]},
        ]
        result = DiversityKCenterSelector(seed=42).select(samples, k=2)
        assert len(result) == 2
        for r in result:
            assert "meta" in r
            assert "min_distance" in r["meta"]

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "embedding": [1.0, 0.0]}]
        result = DiversityKCenterSelector().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = DiversityKCenterSelector().select([], k=3)
        assert result == []

    def test_select_no_embedding(self):
        samples = [{"instruction": "a"}, {"instruction": "b"}]
        result = DiversityKCenterSelector().select(samples, k=2)
        assert result == []

    def test_select_k_larger_than_valid(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b"},
        ]
        result = DiversityKCenterSelector(seed=42).select(samples, k=2)
        assert len(result) == 1

    def test_deterministic_with_seed(self):
        samples = [
            {"instruction": str(i), "embedding": [float(i), 0.0]} for i in range(10)
        ]
        a = DiversityKCenterSelector(seed=42).select(samples, k=3)
        b = DiversityKCenterSelector(seed=42).select(samples, k=3)
        assert [s["instruction"] for s in a] == [s["instruction"] for s in b]
