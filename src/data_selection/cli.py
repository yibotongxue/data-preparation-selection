from __future__ import annotations

import argparse
import importlib

from data_selection.config import CustomOmegaConfig
from data_selection.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select",
        description="Run data selection on a JSONL file via a Python config.",
    )
    parser.add_argument("--config", required=True, help="Python module path.")
    parser.add_argument("--input", default=None, help="Override input JSONL path.")
    parser.add_argument("--output", default=None, help="Override output JSONL path.")
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
    args = parser.parse_args()

    module = importlib.import_module(args.config)
    dataset_cfg = module.dataset().create()
    selector_cfg: CustomOmegaConfig = module.selector()["selector"]

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

    samples = read_jsonl(dataset_cfg.path, dataset_cfg)
    selector = selector_cfg.create()
    selected = selector.select(samples)

    write_jsonl(selected, dataset_cfg.output)
    print(f"Selected {len(selected)} samples -> {dataset_cfg.output}")


if __name__ == "__main__":
    main()
