from __future__ import annotations

import argparse
import importlib

from data_selection._resolve import resolve_target
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
    raw: dict = module.selector()

    input_path: str = raw["input"]
    output_path: str = raw["output"]
    selector_cfg = raw["selector"]

    samples = read_jsonl(input_path)
    selector = resolve_target(selector_cfg)
    assert hasattr(selector, "select")
    selected = selector.select(samples)  # type: ignore[attr-defined,union-attr]
    write_jsonl(selected, output_path)
    print(f"Selected {len(selected)} samples -> {output_path}")


if __name__ == "__main__":
    main()
