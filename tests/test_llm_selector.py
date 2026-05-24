from data_selection.llm_selector import LLMAsSelector, _fallback_score


class FakeLLMClient:
    """Fake client that scores by text length."""

    def score(self, samples: list[dict]) -> list[float]:
        result = []
        for s in samples:
            text = str(s.get("instruction", "")) + " " + str(s.get("output", ""))
            result.append(float(len(text)))
        return result


class TestLLMAsSelector:
    def test_select_with_client(self):
        samples = [
            {"instruction": "x", "output": "short"},
            {"instruction": "x", "output": "a" * 100},
            {"instruction": "x", "output": "medium text"},
        ]
        client = FakeLLMClient()
        result = LLMAsSelector(client=client).select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[1]

    def test_select_fallback(self):
        samples = [
            {"instruction": "q", "output": "hi"},
            {"instruction": "q", "output": "hello world " * 10},
            {"instruction": "q", "output": "ok"},
        ]
        result = LLMAsSelector().select(samples, k=2)
        assert len(result) == 2
        assert result[0] == samples[1]

    def test_select_k_zero(self):
        result = LLMAsSelector().select([{"instruction": "hi", "output": "x"}], k=0)
        assert result == []

    def test_select_empty(self):
        result = LLMAsSelector().select([], k=3)
        assert result == []

    def test_fallback_score_instruction_output(self):
        sample = {"instruction": "x y", "output": "a b c"}
        score = _fallback_score(sample)
        assert score == 5.0

    def test_fallback_score_conversations(self):
        sample = {
            "conversations": [
                {
                    "messages": [
                        {"content": "hello world"},
                    ]
                }
            ]
        }
        score = _fallback_score(sample)
        assert score == 2.0
