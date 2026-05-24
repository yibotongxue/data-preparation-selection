from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from data_selection.selectors.diversity_kcenter import (
    DiversityKCenterSelector,
)


class TestDiversityKCenterSelection:
    def test_select_basic(self, monkeypatch):
        mock_tsds = MagicMock()
        monkeypatch.setattr(
            "data_selection.selectors.diversity_kcenter.offline_tsds_Selector",
            mock_tsds,
        )
        probs = np.array([0.1, 0.9, 0.5])
        instance = mock_tsds.return_value
        instance.selector.return_value = probs

        samples = [
            {"instruction": "a", "output": "x"},
            {"instruction": "b", "output": "y"},
            {"instruction": "c", "output": "z"},
        ]
        result = DiversityKCenterSelector(k=2).select(samples)
        assert len(result) == 2
        assert result[0]["instruction"] == "b"
        assert result[0]["meta"]["tsds_probability"] == 0.9

    def test_select_k_zero(self):
        assert DiversityKCenterSelector(k=0).select([{"instruction": "a"}]) == []

    def test_select_empty(self):
        assert DiversityKCenterSelector(k=3).select([]) == []
