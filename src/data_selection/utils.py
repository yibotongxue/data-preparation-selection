import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read samples from a JSONL file."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(samples: list[dict], path: str | Path) -> None:
    """Write samples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def extract_text(sample: dict) -> str:
    """Extract text content from a sample in unified format.

    Supports two formats:
      - instruction/output: {"instruction": "...", "output": "..."}
      - conversations: {"conversations": [{"messages": [...]}]}
    """
    if "conversations" in sample:
        conversations = sample["conversations"]
        if isinstance(conversations, list):
            parts = []
            for turn in conversations:
                messages = turn.get("messages", []) if isinstance(turn, dict) else []
                for msg in messages:
                    parts.append(str(msg.get("content", "")))
            return "\n".join(parts)

    parts = []
    for key in ("instruction", "output"):
        if key in sample:
            parts.append(str(sample[key]))
    return "\n".join(parts)
