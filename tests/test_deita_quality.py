from data_selection.deita_quality import DeitaQualitySelection


class TestDeitaQualitySelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "a", "quality_score": 0.8, "complexity_score": 0.9},
            {"instruction": "b", "quality_score": 0.3, "complexity_score": 0.4},
            {"instruction": "c", "quality_score": 0.9, "complexity_score": 0.5},
        ]
        result = DeitaQualitySelection().select(samples, k=2)
        assert len(result) == 2
        assert result[0]["instruction"] == "a"
        assert result[1]["instruction"] == "c"

    def test_select_k_zero(self):
        samples = [{"instruction": "a", "quality_score": 0.5, "complexity_score": 0.5}]
        result = DeitaQualitySelection().select(samples, k=0)
        assert result == []

    def test_select_empty(self):
        result = DeitaQualitySelection().select([], k=3)
        assert result == []

    def test_select_missing_scores(self):
        samples = [
            {"instruction": "a"},
            {"instruction": "b", "quality_score": 0.5},
        ]
        result = DeitaQualitySelection().select(samples, k=2)
        assert len(result) == 2

    def test_custom_keys(self):
        samples = [
            {"instruction": "a", "q": 0.9, "c": 0.8},
            {"instruction": "b", "q": 0.1, "c": 0.2},
        ]
        selector = DeitaQualitySelection(quality_key="q", complexity_key="c")
        result = selector.select(samples, k=1)
        assert result[0]["instruction"] == "a"
