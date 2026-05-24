from abc import ABC, abstractmethod


class SelectionMethod(ABC):
    """Protocol for data selection methods.

    Each concrete implementation selects a subset of k samples from a
    candidate pool according to a specific strategy.
    """

    @abstractmethod
    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        """Select top-k samples from the candidate pool.

        Args:
            samples: List of candidate samples, each a dict with at least
                     enough fields for the specific selection strategy.
            k: Number of samples to select.
            **kwargs: Strategy-specific parameters.

        Returns:
            The selected subset of samples (length ≤ k).
        """
        ...
