from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from data_selection.config import MaybeConfig, maybe_create
from data_selection.protocol import Selector


class CompositeSelector:
    """Chain multiple selectors, each filtering the output of the previous."""

    def __init__(
        self,
        selectors: list[MaybeConfig[Selector]],
        k: int = 100,
    ) -> None:
        self._selectors: list[Selector] = [maybe_create(s) for s in selectors]  # type: ignore[misc]
        self.k = k

    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = list(samples)  # type: ignore[arg-type]
        for sel in self._selectors:
            result = sel.select(result)
        return result
