from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """A selectable downstream model, ordered from least to most capable."""

    key: str
    provider_model: str
    description: str
    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0


@dataclass(frozen=True)
class Judgment:
    """The reusable structured signals returned by Jev for one request."""

    choice: str
    confidence: float
    probabilities: Dict[str, float]
    deep_reasoning: float
    blast_radius: float
    ambiguity: float
    input_tokens: int = 0
    latency_ms: int = 0
    raw_response: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Decision:
    model: ModelSpec
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    judgment: Optional[Judgment] = None
    jev_available: bool = True

    @property
    def why(self) -> str:
        return " -> ".join(self.reasons)
