from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Ensure project root (where configs/ lives) is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_selection.config import CustomOmegaConfig
from data_selection.utils import read_jsonl, write_jsonl

# Selectors that produce deterministic ranked results — running once with max(k)
# and truncating is equivalent to running separately for each k.
_SCORE_BASED_SELECTORS = {
    "LengthBasedSelector",
    "PerplexityBasedSelector",
    "DeitaQualitySelector",
    "DeitaQualityRaySelector",
    "QualityScorerSelector",
    "EmbeddingSimilaritySelector",
    "DiversityKCenterSelector",
    "LLMAsSelector",
}

# Selectors that support scores_cache_path for resume
_CACHEABLE_SELECTORS = {
    "PerplexityBasedSelector",
    "DeitaQualitySelector",
    "DeitaQualityRaySelector",
    "QualityScorerSelector",
    "LLMAsSelector",
}


def _resolve_output_path(template: str, k: int) -> str:
    """Replace {k} placeholder in output path, or insert _k before extension."""
    if "{k}" in template:
        return template.replace("{k}", str(k))
    base, ext = os.path.splitext(template)
    return f"{base}_{k}{ext}"


def _get_selector_name(selector_cfg: CustomOmegaConfig) -> str:
    target: str = selector_cfg.get("_target_", "")
    return target.rsplit(".", 1)[-1] if target else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select",
        description="Run data selection on a JSONL file via a Python config.",
    )
    parser.add_argument(
        "--config", required=True, help="Python module path, e.g. configs.random"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Override k. Multiple values supported (e.g. --k 1000 5000 10000).",
    )
    parser.add_argument("--input", default=None, help="Override the input path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output path. Use {k} as placeholder for multi-k mode.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for score caches (enables resume).",
    )
    parser.add_argument(
        "--instruction-key", default=None, help="Override alpaca instruction key."
    )
    parser.add_argument(
        "--output-key", default=None, help="Override alpaca output key."
    )
    parser.add_argument(
        "--conversations-key",
        default=None,
        help="Override sharegpt/openai key.",
    )
    parser.add_argument(
        "--query-path",
        default=None,
        help="JSONL target set for EmbeddingSimilaritySelector (NEAR query).",
    )
    parser.add_argument(
        "--query-key",
        default=None,
        help="Field name holding the query text in --query-path (default: query).",
    )
    args = parser.parse_args()

    module = importlib.import_module(args.config)

    # dataset() provides input/output path and key names
    dataset_cfg = module.dataset().create()

    # selector() returns {"selector": CustomOmegaConfig, ...}
    selector_cfg: CustomOmegaConfig = module.selector()["selector"]

    # Apply CLI overrides
    if args.input:
        dataset_cfg.path = args.input
    if args.output:
        dataset_cfg.output = args.output
    if args.instruction_key:
        dataset_cfg.instruction_key = args.instruction_key
    if args.output_key:
        dataset_cfg.output_key = args.output_key
    if args.conversations_key:
        dataset_cfg.conversations_key = args.conversations_key

    # Inject EmbeddingSimilarity target-set (NEAR query) overrides if provided
    if args.query_path:
        selector_cfg["query_path"] = args.query_path
    if args.query_key:
        selector_cfg["query_key"] = args.query_key

    # Detect selector type
    selector_name = _get_selector_name(selector_cfg)
    is_score_based = selector_name in _SCORE_BASED_SELECTORS

    # Inject scores_cache_path if --cache-dir is specified and selector supports it
    if args.cache_dir and selector_name in _CACHEABLE_SELECTORS:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(args.cache_dir, f"{selector_name}_scores.jsonl")
        selector_cfg["scores_cache_path"] = cache_path
        print(f"[Resume] Using score cache: {cache_path}")

    # Read samples
    samples = read_jsonl(dataset_cfg.path, dataset_cfg)

    # Determine k values
    if args.k is not None:
        k_values: list[int] = args.k
    else:
        cfg_k = selector_cfg.get("k", 100)
        k_values = list(cfg_k) if isinstance(cfg_k, (list, tuple)) else [cfg_k]

    if is_score_based and len(k_values) > 1:
        # Score-based: run once with max(k), then truncate for each k
        max_k = max(k_values)
        selector_cfg.k = max_k
        selector = selector_cfg.create()
        all_selected = selector.select(samples)

        for k in sorted(k_values):
            out = _resolve_output_path(dataset_cfg.output, k)
            truncated = all_selected[:k]
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write_jsonl(truncated, out)
            print(f"Selected {len(truncated)} samples -> {out}")
    else:
        # Random-based or single k: run independently for each k
        for k in k_values:
            selector_cfg.k = k
            selector = selector_cfg.create()
            selected = selector.select(samples)

            if len(k_values) > 1:
                out = _resolve_output_path(dataset_cfg.output, k)
            else:
                out = dataset_cfg.output

            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            write_jsonl(selected, out)
            print(f"Selected {len(selected)} samples -> {out}")


if __name__ == "__main__":
    main()
