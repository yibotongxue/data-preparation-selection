from data_selection.utils import extract_text


class LengthBasedSelection:
    """Select samples by instruction+output or conversations length."""

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        def _length(s: dict) -> int:
            return len(extract_text(s))

        scored = sorted(samples, key=_length, reverse=True)
        return scored[: min(k, len(scored))]
