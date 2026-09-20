"""The model pool and every policy threshold live in one reviewable module."""

import os
from dataclasses import dataclass
from typing import Tuple

from .models import ModelSpec


# Change provider_model values to the model IDs enabled in your OpenRouter account.
# Ordering is capability ordering; policy moves only toward the right of this list.
DEFAULT_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec("fast", "openai/gpt-4.1-mini", "Mechanical transformations and bounded factual work."),
    ModelSpec("balanced", "anthropic/claude-sonnet-4", "General professional work and bounded implementation."),
    ModelSpec("frontier", "openai/gpt-5", "Complex design, diagnosis, high-risk, or ambiguous work."),
    ModelSpec("extended", "anthropic/claude-opus-4", "Large-context and long autonomous synthesis."),
)


@dataclass(frozen=True)
class Thresholds:
    min_confidence: float = 0.55
    reasoning_floor: str = "balanced"
    deep_reasoning_at: float = 0.70
    deep_reasoning_escalate_at: float = 0.87
    ambiguity_at: float = 0.80
    blast_radius_at: float = 1.0
    risk_ceiling: str = "frontier"
    fallback: str = "balanced"


THRESHOLDS = Thresholds()
JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")
