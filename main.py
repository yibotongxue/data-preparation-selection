from data_selection import RandomSelection


def main() -> None:
    samples = [
        {"text": "sample 1", "source": "a"},
        {"text": "sample 2", "source": "b"},
        {"text": "sample 3", "source": "a"},
        {"text": "sample 4", "source": "b"},
        {"text": "sample 5", "source": "c"},
    ]
    selector = RandomSelection(seed=42)
    selected = selector.select(samples, k=3)
    for s in selected:
        print(s["text"])


if __name__ == "__main__":
    main()
