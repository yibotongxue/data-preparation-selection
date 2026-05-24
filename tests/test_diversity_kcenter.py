from __future__ import annotations

from data_selection.selectors.diversity_kcenter import DiversityKCenterSelector


class TestDiversityKCenterSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "a", "embedding": [1.0, 0.0]},
            {"instruction": "b", "embedding": [0.0, 1.0]},
            {"instruction": "c", "embedding": [1.0, 0.1]},
            {"instruction": "d", "embedding": [0.0, 0.9]},
        ]
        result = DiversityKCenterSelector(k=2, seed=42).select(samples)
        assert len(result) == 2
        for r in result:
            assert "meta" in r
            assert "min_distance" in r["meta"]

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "embedding": [1.0, 0.0]}]
        result = DiversityKCenterSelector(k=0).select(samples)
        assert result == []

    def test_select_empty(self):
        result = DiversityKCenterSelector(k=3).select([])
        assert result == []

    def test_select_no_embedding(self):
        result = DiversityKCenterSelector(k=2).select(
            [{"instruction": "a"}, {"instruction": "b"}]
        )
        assert result == []

    def test_deterministic_with_seed(self):
        samples = [
            {"instruction": str(i), "embedding": [float(i), 0.0]} for i in range(10)
        ]
        a = DiversityKCenterSelector(k=3, seed=42).select(samples)
        b = DiversityKCenterSelector(k=3, seed=42).select(samples)
        assert [s["instruction"] for s in a] == [s["instruction"] for s in b]
