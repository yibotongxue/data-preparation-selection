import argparse
import importlib
import inspect
import json
import sys
from typing import Any, get_args, get_origin

from data_selection.config import CustomOmegaConfig
from data_selection.selectors import (
    DeitaQualitySelector,
    DiversityKCenterSelector,
    EmbeddingSimilaritySelector,
    LengthBasedSelector,
    LLMAsSelector,
    PerplexityBasedSelector,
    QualityScorerSelector,
    RandomSelector,
    SourceBalancedRandomSelector,
)
from data_selection.utils import read_jsonl, write_jsonl

SELECTOR_MAP: dict[str, type] = {
    "random": RandomSelector,
    "source_balanced_random": SourceBalancedRandomSelector,
    "length_based": LengthBasedSelector,
    "perplexity_based": PerplexityBasedSelector,
    "embedding_similarity": EmbeddingSimilaritySelector,
    "deita_quality": DeitaQualitySelector,
    "quality_scorer": QualityScorerSelector,
    "diversity_kcenter": DiversityKCenterSelector,
    "llm_selector": LLMAsSelector,
}

SKIP_PARAMS = {
    "scorer",
    "quality_scorer",
    "complexity_scorer",
    "edu_scorer",
    "pq_scorer",
    "llm_serving",
    "dimensions",
    "domain_proxy",
}


def _type_from_annotation(ann: Any) -> type:
    origin = get_origin(ann)
    if origin is not None:
        args = get_args(ann)
        for a in args:
            if a is not type(None):
                return _type_from_annotation(a)
        return str
    if ann is bool:
        return bool
    if ann is int:
        return int
    if ann is float:
        return float
    return str


def _cli_params(selector_cls: type) -> dict[str, dict]:
    sig = inspect.signature(selector_cls.__init__)
    params = {}
    for name, p in sig.parameters.items():
        if name in ("self", "kwargs"):
            continue
        if name in SKIP_PARAMS:
            continue
        ann = p.annotation if p.annotation is not inspect.Parameter.empty else str
        typ = _type_from_annotation(ann)
        default = p.default if p.default is not inspect.Parameter.empty else None
        params[name] = {"type": typ, "default": default}
    return params


def _resolve_method() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--method" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return ""


def main() -> None:
    method = _resolve_method()
    cls = SELECTOR_MAP.get(method)

    parser = argparse.ArgumentParser(
        prog="select",
        description="Run data selection on a JSONL file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Python module path with a selector() function (e.g. my_project.my_config).",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=list(SELECTOR_MAP),
        help="Quick selection method name (alternative to --config).",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of samples to select (default: 100).",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="JSON string of param overrides for --config mode (e.g. '{\"seed\": 42}').",
    )

    if cls is not None:
        for name, info in _cli_params(cls).items():
            flag = f"--{name}"
            parser.add_argument(
                flag,
                type=info["type"],
                default=info["default"],
                help=f"[{cls.__name__}] (default: {info['default']}).",
            )

    args = parser.parse_args()

    samples = read_jsonl(args.input)

    if args.config:
        module = importlib.import_module(args.config)
        config: CustomOmegaConfig = module.selector()
        if args.overrides:
            overrides = json.loads(args.overrides)
            config = config.merge(**overrides)
        selector = config.create()
    elif args.method:
        cls = SELECTOR_MAP[args.method]
        init_kwargs = {}
        for name, info in _cli_params(cls).items():
            val = getattr(args, name)
            if val != info["default"]:
                init_kwargs[name] = val
        selector = cls(**init_kwargs)
    else:
        parser.error("Either --config or --method is required.")

    selected = selector.select(samples, k=args.k)
    write_jsonl(selected, args.output)
    print(f"Selected {len(selected)} samples -> {args.output}")


if __name__ == "__main__":
    main()
