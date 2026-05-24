from data_selection.protocol import Selector


class CompositeSelector:
    """Chain multiple selectors, each filtering the output of the previous."""

    def __init__(self, *selectors: Selector, k: int = 100) -> None:
        self._selectors = list(selectors)
        self.k = k

    def select(self, samples: list[dict]) -> list[dict]:
        result = samples
        for sel in self._selectors:
            result = sel.select(result)
        return result
