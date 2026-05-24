from __future__ import annotations

from typing import Protocol


class Selector(Protocol):
    """Protocol for data selectors.

    All strategy-specific parameters (including k) are configured
    at __init__ time. Any class with a compatible `select` method
    satisfies this protocol.
    """

    def select(self, samples: list[dict]) -> list[dict]:
        """Select samples from the candidate pool.

        Args:
            samples: Candidate samples as a list of dicts.

        Returns:
            The selected subset, each with a "meta" dict attached.
        """
        ...
