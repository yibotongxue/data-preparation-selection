# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a data selection framework for LLM training data. The package name is `data_selection` under `src/`, with Python 3.12 and `uv` as the package manager.

## Commands

```bash
# Install dependencies
uv sync

# Run the entry point
uv run python main.py

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_random_selection.py

# Type checking (pyright)
uv run --with pyright pyright

# Run pre-commit on all files
uv run pre-commit run --all-files
```

## Dependency management

Use `uv add <package>` for all dependencies. Never use `pip`, `uv pip install`, or manually edit `pyproject.toml`.

## Architecture

```
src/data_selection/
├── __init__.py          # Top-level re-export
├── protocol.py          # SelectionMethod Protocol (structural subtyping)
├── utils.py             # extract_text — unified instruction/output or conversations parsing
└── selectors/           # All 9 selection method implementations
    ├── random_selection.py
    ├── source_balanced_random.py
    ├── length_based.py
    ├── perplexity_based.py       # DataFlow PerplexityScorer (Kenlm)
    ├── embedding_similarity.py   # Near algorithm (numpy)
    ├── deita_quality.py          # DataFlow DeitaQualityScorer + DeitaComplexityScorer
    ├── quality_scorer.py         # DataFlow FineWebEduScorer + PairQualScorer
    ├── diversity_kcenter.py      # TSDS algorithm (numpy)
    └── llm_selector.py           # DataFlow MetaScorer
```

### SelectionMethod Protocol

Defined in `protocol.py` — uses `typing.Protocol` for structural subtyping (classes do NOT need to inherit). The interface is:

```python
class SelectionMethod(Protocol):
    def select(self, samples: list[dict], k: int) -> list[dict]: ...
```

All strategy parameters go into `__init__`, not `select()`. The `select()` signature is always `(samples, k)`.

### Input format

Samples must use one of two formats:
- `{"instruction": "...", "output": "..."}`
- `{"conversations": [{"messages": [{"content": "..."}]}]}`

Use `extract_text()` from `utils.py` to parse either format.

### Dependencies

- `numpy` — vectorized math for embedding_similarity and diversity_kcenter
- `open-dataflow` — quality scorers: DeitaQuality/Complexity, FineWebEdu, PairQual, Perplexity(Kenlm), MetaScorer
- `dataflex` (OpenDCAI/DataFlex) is NOT installed — it requires `numpy<2.0`, incompatible with `open-dataflow`. Near and TSDS algorithms are implemented directly with numpy.

### Testing

Tests use `unittest.mock.MagicMock` for DataFlow scorer injection. DataFlow-based selectors accept optional scorer params in `__init__` for testability:

```python
mock = MagicMock()
mock.eval.return_value = [1.0, 2.0, 3.0]
selector = DeitaQualitySelection(quality_scorer=mock, complexity_scorer=mock)
```

## Pre-commit pipeline

`isort` → `black-jupyter` → `autoflake` → `pyupgrade` → `bandit` (security, configured in `.bandit.yml`) → `pyright` (type check) → `codespell` (spelling). Some checks modify files in-place — re-stage after a failed commit.

## Config

- `pyproject.toml`: Package metadata, Python `>=3.12,<3.13`, black (line-length 88), isort (black profile).
- `.bandit.yml`: Excludes `.venv`, `venv`, `build`, `dist`; medium severity/confidence.
