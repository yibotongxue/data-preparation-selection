import argparse
import importlib

from data_selection.config import CustomOmegaConfig
from data_selection.types import SelectionConfig
from data_selection.utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select",
        description="Run data selection on a JSONL file via a Python config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path with a selector() function (e.g. my_project.my_config).",
    )
    args = parser.parse_args()

    module = importlib.import_module(args.config)
    raw: SelectionConfig | dict = module.selector()

    input_path: str = raw["input"]
    output_path: str = raw["output"]
    selector_cfg: CustomOmegaConfig = raw["selector"]

    samples = read_jsonl(input_path)
    selector = selector_cfg.create()
    selected = selector.select(samples)
    write_jsonl(selected, output_path)
    print(f"Selected {len(selected)} samples -> {output_path}")


if __name__ == "__main__":
    main()
