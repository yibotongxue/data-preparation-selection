from typing import Protocol


class Selector(Protocol):
    """Protocol for data selection methods.

    Each concrete implementation selects a subset of k samples from a
    candidate pool according to a specific strategy. All strategy-specific
    parameters are configured at __init__ time.

    Any class with a compatible `select` method satisfies this protocol.
    """

    def select(self, samples: list[dict], k: int) -> list[dict]:
        """Select top-k samples from the candidate pool.

        Args:
            samples: List of candidate samples, each a dict with
                     instruction/output or conversations fields.
            k: Number of samples to select.

        Returns:
            The selected subset of samples (length ≤ k).
        """
        ...
