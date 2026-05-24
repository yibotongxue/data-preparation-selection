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
