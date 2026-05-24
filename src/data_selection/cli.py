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
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path with a selector() function.",
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

    samples = read_jsonl(input_path)
    selector = selector_cfg.create()
    selected = selector.select(samples)  # type: ignore[union-attr]
    write_jsonl(selected, output_path)
    print(f"Selected {len(selected)} samples -> {output_path}")


if __name__ == "__main__":
    main()
