class CompositeSelector:
    """Chain multiple selectors, each filtering the output of the previous."""

    def __init__(self, selectors: list) -> None:
        self._selectors = selectors

    def select(self, samples: list[dict], k: int) -> list[dict]:
        result = samples
        for sel in self._selectors:
            result = sel.select(result, k)
        return result
