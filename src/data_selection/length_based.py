from data_selection.protocol import SelectionMethod


class LengthBasedSelection(SelectionMethod):
    """Select samples by prompt/response length as a proxy for information content."""

    def __init__(
        self,
        prompt_key: str = "prompt",
        response_key: str = "response",
        text_key: str | None = None,
    ) -> None:
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.text_key = text_key

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        def _length(s: dict) -> int:
            if self.text_key:
                return len(str(s.get(self.text_key, "")))
            return len(str(s.get(self.prompt_key, ""))) + len(
                str(s.get(self.response_key, ""))
            )

        scored = sorted(samples, key=_length, reverse=True)
        return scored[: min(k, len(scored))]
