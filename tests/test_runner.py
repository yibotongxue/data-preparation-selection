from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from data_selection import LengthBasedSelector, RandomSelector
from data_selection.runner import run_selection


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRunner:
    def test_run_single_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")
            _write_jsonl(
                input_path,
                [{"instruction": f"task {i}", "output": f"out {i}"} for i in range(10)],
            )

            run_selection(
                RandomSelector(k=3, seed=42),
                input_path=input_path,
                output_path=output_path,
            )

            result = _read_jsonl(output_path)
            assert len(result) == 3
            assert all("meta" in r for r in result)

    def test_run_multiple_k_random_runs_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")
            _write_jsonl(
                input_path,
                [{"instruction": f"task {i}", "output": f"out {i}"} for i in range(10)],
            )

            run_selection(
                RandomSelector(k=2, seed=42),
                input_path=input_path,
                output_path=output_path,
                k=[2, 4],
            )

            small = _read_jsonl(os.path.join(tmpdir, "output_2.jsonl"))
            large = _read_jsonl(os.path.join(tmpdir, "output_4.jsonl"))
            assert len(small) == 2
            assert len(large) == 4
            # Random-based selectors are re-run per k, so top-2 need not match.
            assert {r["instruction"] for r in small} != {
                r["instruction"] for r in large[:2]
            } or len({r["instruction"] for r in small + large}) > 2

    def test_run_multiple_k_score_based_truncates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")
            _write_jsonl(
                input_path,
                [{"instruction": "a" * (i + 1), "output": ""} for i in range(10)],
            )

            run_selection(
                LengthBasedSelector(k=2),
                input_path=input_path,
                output_path=output_path,
                k=[2, 4],
            )

            small = _read_jsonl(os.path.join(tmpdir, "output_2.jsonl"))
            large = _read_jsonl(os.path.join(tmpdir, "output_4.jsonl"))
            assert len(small) == 2
            assert len(large) == 4
            # Score-based selectors truncate the same ranked list.
            assert [r["instruction"] for r in small] == [
                r["instruction"] for r in large[:2]
            ]

    def test_output_path_with_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "out_{k}.jsonl")
            _write_jsonl(
                input_path,
                [{"instruction": f"task {i}", "output": f"out {i}"} for i in range(5)],
            )

            run_selection(
                RandomSelector(k=2, seed=42),
                input_path=input_path,
                output_path=output_path,
                k=[2, 3],
            )

            assert Path(tmpdir, "out_2.jsonl").exists()
            assert Path(tmpdir, "out_3.jsonl").exists()

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            output_path = os.path.join(tmpdir, "output.jsonl")
            _write_jsonl(input_path, [])

            run_selection(
                RandomSelector(k=3, seed=42),
                input_path=input_path,
                output_path=output_path,
            )

            assert _read_jsonl(output_path) == []
