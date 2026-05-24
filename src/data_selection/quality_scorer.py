import re

from data_selection.utils import extract_text


class QualityScorerSelection:
    """Select samples by educational/quality score (FineWeb-Edu / PairQual style).

    Uses heuristics to estimate educational value:
      - sentence_length: longer sentences suggest more structured content
      - vocabulary_richness: unique word ratio
      - format_quality: presence of structured formatting (lists, headers, code)
    """

    def select(self, samples: list[dict], k: int, **kwargs) -> list[dict]:
        if k <= 0 or not samples:
            return []

        scored = []
        for s in samples:
            score = _edu_score(s)
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: min(k, len(scored))]]


def _edu_score(sample: dict) -> float:
    text = extract_text(sample)
    if not text:
        return 0.0

    sentences = re.split(r"[.!?]+", text)
    sentences = [st.strip() for st in sentences if st.strip()]
    if not sentences:
        return 0.0

    words = text.lower().split()
    unique_words = set(words)
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    vocab_richness = len(unique_words) / max(len(words), 1)

    formatting_score = 0.0
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE):
        formatting_score += 0.2
    if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
        formatting_score += 0.2
    if "```" in text:
        formatting_score += 0.2

    return (
        0.3 * min(avg_sentence_len / 30, 1.0) + 0.4 * vocab_richness + formatting_score
    )
