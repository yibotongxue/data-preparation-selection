from data_selection.diversity_kcenter import DiversityKCenterSelection


class TestDiversityKCenterSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 1.0]},
            {"instruction": "c", "embedding": [1.0, 0.1]},
            {"instruction": "d", "embedding": [0.0, 0.9]},
        ]
        result = DiversityKCenterSelection(seed=42).select(samples, k=2)
        assert len(result) == 2

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "embedding": [1.0, 0.0]}]
        result = DiversityKCenterSelection().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = DiversityKCenterSelection().select([], k=3)
        assert result == []

    def test_select_no_embedding(self):
        samples = [{"instruction": "a"}, {"instruction": "b"}]
        result = DiversityKCenterSelection().select(samples, k=2)
        assert result == []

    def test_select_k_larger_than_valid(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b"},
        ]
        result = DiversityKCenterSelection(seed=42).select(samples, k=2)
        assert len(result) == 1

    def test_deterministic_with_seed(self):
        samples = [
            {"instruction": str(i), "embedding": [float(i), 0.0]} for i in range(10)
        ]
        a = DiversityKCenterSelection(seed=42).select(samples, k=3)
        b = DiversityKCenterSelection(seed=42).select(samples, k=3)
        assert [s["instruction"] for s in a] == [s["instruction"] for s in b]
