from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from data_selection.config import CustomOmegaConfig
    from data_selection.protocol import Selector


class Message(TypedDict):
    content: str


class Turn(TypedDict):
    messages: list[Message]


class Sample(TypedDict, total=False):
    instruction: str
    output: str
    conversations: list[Turn]
    source: str
    embedding: list[float]
    ppl: float
    quality_score: float
    complexity_score: float
    text: str
    meta: Meta


class Meta(TypedDict, total=False):
    selector: str
    seed: NotRequired[int | None]
    source: NotRequired[str]
    length: NotRequired[int]
    ppl: NotRequired[float]
    strategy: NotRequired[str]
    similarity: NotRequired[float]
    quality_score: NotRequired[float]
    complexity_score: NotRequired[float]
    composite_score: NotRequired[float]
    fineweb_edu_score: NotRequired[float | None]
    pairqual_score: NotRequired[float | None]
    min_distance: NotRequired[float]
    dimension_scores: NotRequired[list[float] | None]
    average_score: NotRequired[float]


class SelectionConfig(TypedDict):
    input: str
    output: str
    k: int
    selector: CustomOmegaConfig[Selector]
