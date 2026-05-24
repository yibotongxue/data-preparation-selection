from typing import Any

from data_selection.protocol import SelectionMethod


class LLMAsSelector(SelectionMethod):
    """Use an LLM to score and rank samples by trainability/relevance/quality.

    This is a stub that expects an external LLM client to be provided.
    The client should have a `score(samples: list[dict]) -> list[float]` method.
    """

    def __init__(self, client: Any = None) -> None:
        """Initialize with an optional LLM client.

        Args:
            client: An object with a `score(samples) -> list[float]` method.
                    If None, falls back to a heuristic placeholder.
        """
        self.client = client

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        if self.client is not None and hasattr(self.client, "score"):
            scores = self.client.score(samples)
        else:
            scores = [self._fallback_score(s) for s in samples]

        paired = list(zip(scores, samples, strict=True))
        paired.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in paired[: min(k, len(paired))]]

    @staticmethod
    def _fallback_score(sample: dict) -> float:
        """Heuristic fallback when no LLM client is available."""
        text = str(sample.get("text", sample.get("response", sample.get("prompt", ""))))
        return float(len(text.split()))
