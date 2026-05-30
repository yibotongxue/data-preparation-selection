from __future__ import annotations

import argparse
import importlib
import os
import sys

# Ensure cwd is on sys.path so configs/ can be imported as a package.
_cwd = os.getcwd()
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from data_selection.config import CustomOmegaConfig
from data_selection.utils import read_jsonl, write_jsonl

# Selectors that produce deterministic ranked results — running once with max(k)
# and truncating is equivalent to running separately for each k.
_SCORE_BASED_SELECTORS = {
    "LengthBasedSelector",
    "PerplexityBasedSelector",
    "DeitaQualitySelector",
    "QualityScorerSelector",
    "EmbeddingSimilaritySelector",
    "DiversityKCenterSelector",
    "LLMAsSelector",
}

# Selectors that support scores_cache_path for resume
_CACHEABLE_SELECTORS = {
    "PerplexityBasedSelector",
    "DeitaQualitySelector",
    "QualityScorerSelector",
    "LLMAsSelector",
}


def _resolve_output_path(template: str, k: int) -> str:
    """Replace {k} placeholder in output path, or insert _k before extension."""
    if "{k}" in template:
        return template.replace("{k}", str(k))
    base, ext = os.path.splitext(template)
    return f"{base}_{k}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select",
        description="Run data selection on a JSONL file via a Python config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path with a selector() function.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Override k. Multiple values supported (e.g. --k 1000 5000 10000).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override the input path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output path. Use {k} as placeholder for multi-k mode.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for score caches (enables resume). "
        "Each selector gets its own cache file: <cache-dir>/<selector_name>_scores.jsonl",
    )
    args = parser.parse_args()

    module = importlib.import_module(args.config)
    raw = module.selector()

    if isinstance(raw, CustomOmegaConfig):
        cfg = raw
        input_path: str = cfg["input"]  # type: ignore[assignment]
        output_path: str = cfg["output"]  # type: ignore[assignment]
        selector_cfg: CustomOmegaConfig = cfg
    else:
        input_path = raw["input"]
        output_path = raw["output"]
        selector_cfg = raw["selector"]

    # Override input if specified
    if args.input is not None:
        input_path = args.input

    # Override output if specified
    if args.output is not None:
        output_path = args.output

    # Detect selector type from _target_
    target: str = selector_cfg.get("_target_", "")
    selector_name = target.rsplit(".", 1)[-1] if target else ""
    is_score_based = selector_name in _SCORE_BASED_SELECTORS

    # Inject scores_cache_path if --cache-dir is specified and selector supports it
    if args.cache_dir and selector_name in _CACHEABLE_SELECTORS:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(args.cache_dir, f"{selector_name}_scores.jsonl")
        selector_cfg["scores_cache_path"] = cache_path
        print(f"[Resume] Using score cache: {cache_path}")

    samples = read_jsonl(input_path)

    # Determine k values
    k_values: list[int] = args.k if args.k is not None else [selector_cfg.get("k", 100)]

    if is_score_based and len(k_values) > 1:
        # Score-based: run once with max(k), then truncate for each k
        max_k = max(k_values)
        selector_cfg["k"] = max_k
        selector = selector_cfg.create()
        all_selected = selector.select(samples)  # type: ignore[union-attr]

        for k in sorted(k_values):
            out = _resolve_output_path(output_path, k)
            truncated = all_selected[:k]
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write_jsonl(truncated, out)
            print(f"Selected {len(truncated)} samples -> {out}")
    else:
        # Random-based or single k: run independently for each k
        for k in k_values:
            selector_cfg["k"] = k
            selector = selector_cfg.create()
            selected = selector.select(samples)  # type: ignore[union-attr]

            if len(k_values) > 1:
                out = _resolve_output_path(output_path, k)
            else:
                out = output_path

            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write_jsonl(selected, out)
            print(f"Selected {len(selected)} samples -> {out}")


if __name__ == "__main__":
    main()
