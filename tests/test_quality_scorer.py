from data_selection.quality_scorer import QualityScorerSelection, _edu_score


class TestQualityScorerSelection:
    def test_select_basic(self):
        samples = [
            {"instruction": "hi", "output": "ok"},
            {
                "instruction": "Explain quantum mechanics",
                "output": "This is a well-structured sentence with multiple words. "
                "It contains educational content and proper formatting. "
                "Here is another sentence with valuable information.",
            },
            {"instruction": "hi", "output": "ok"},
        ]
        result = QualityScorerSelection().select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[1]

    def test_select_k_zero(self):
        result = QualityScorerSelection().select(
            [{"instruction": "hi", "output": "there"}], k=0
        )
        assert result == []

    def test_select_empty(self):
        result = QualityScorerSelection().select([], k=3)
        assert result == []

    def test_select_empty_text(self):
        samples = [
            {"instruction": "", "output": ""},
            {"instruction": "proper", "output": "some content here"},
        ]
        result = QualityScorerSelection().select(samples, k=2)
        assert len(result) == 2

    def test_formatting_bonus(self):
        simple = {"instruction": "say hi", "output": "just plain text here."}
        formatted = {
            "instruction": "write code",
            "output": "```\ncode block\n```\n- item 1\n- item 2\n1. step one\n2. step two",
        }
        score_simple = _edu_score(simple)
        score_formatted = _edu_score(formatted)
        assert score_formatted > score_simple

    def test_conversations_format(self):
        sample = {
            "conversations": [
                {
                    "messages": [
                        {"content": "What is Python?"},
                        {"content": "Python is a programming language."},
                    ]
                }
            ]
        }
        score = _edu_score(sample)
        assert score > 0.0
