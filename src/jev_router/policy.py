"""Pure routing policy: deterministic, offline-testable, and conservative."""

from typing import Iterable, Optional, Sequence, Tuple

from .config import THRESHOLDS, Thresholds
from .models import Decision, Judgment, ModelSpec


def _index(models: Sequence[ModelSpec], key: str) -> int:
    for index, model in enumerate(models):
        if model.key == key:
            return index
    raise ValueError("unknown model key %r; expected one of %r" % (key, tuple(m.key for m in models)))


def _allowed(models: Sequence[ModelSpec], allowed: Optional[Iterable[str]]) -> Tuple[ModelSpec, ...]:
    selected = tuple(models) if allowed is None else tuple(m for m in models if m.key in set(allowed))
    if not selected:
        raise ValueError("allowed models cannot be empty")
    return selected


def _settle(models, rank, allowed, min_model, max_model):
    low = _index(models, min_model) if min_model else 0
    high = _index(models, max_model) if max_model else len(models) - 1
    if low > high:
        raise ValueError("min_model is more capable than max_model")
    rank = max(low, min(rank, high))
    candidates = _allowed(models[low : high + 1], allowed)
    # A disallowed choice climbs first; only a hard cap can force a move down.
    above = [m for m in candidates if _index(models, m.key) >= rank]
    return above[0] if above else candidates[-1]


def decide(
    models: Sequence[ModelSpec], judgment: Optional[Judgment], *, force_model: Optional[str] = None,
    allowed: Optional[Iterable[str]] = None, min_model: Optional[str] = None,
    max_model: Optional[str] = None, thresholds: Thresholds = THRESHOLDS,
) -> Decision:
    """Convert Jev signals into exactly one downstream model.

    Jev never has permission to bypass provider caps: operator constraints are
    deterministic controls, and uncertainty never causes a cheaper route.
    """
    if not models:
        raise ValueError("models cannot be empty")
    reasons = []
    if force_model:
        rank = _index(models, force_model)
        reasons.append("forced:%s" % force_model)
        model = _settle(models, rank, allowed, min_model, max_model)
        if model.key != force_model:
            reasons.append("clamped:%s" % model.key)
        return Decision(model, tuple(reasons), judgment, judgment is not None)

    valid_choices = {model.key for model in models}
    if judgment is None or judgment.choice not in valid_choices:
        rank = _index(models, thresholds.fallback)
        reasons.append("jev-unavailable:conservative-fallback")
        return Decision(_settle(models, rank, allowed, min_model, max_model), tuple(reasons), judgment, False)

    rank = _index(models, judgment.choice)
    reasons.append("jev:%s@%.2f" % (judgment.choice, judgment.confidence))
    floor = _index(models, thresholds.reasoning_floor)
    if judgment.confidence < thresholds.min_confidence and rank < floor:
        rank = floor
        reasons.append("low-confidence-floor:%s" % thresholds.reasoning_floor)
    if judgment.deep_reasoning > thresholds.deep_reasoning_at and rank < floor:
        rank = floor
        reasons.append("reasoning-floor:%s" % thresholds.reasoning_floor)

    bumps = 0
    if judgment.deep_reasoning > thresholds.deep_reasoning_escalate_at:
        bumps += 1
        reasons.append("hard-reasoning:+1")
    if judgment.ambiguity > thresholds.ambiguity_at:
        bumps += 1
        reasons.append("ambiguous:+1")
    if judgment.blast_radius > thresholds.blast_radius_at:
        bumps += 1
        reasons.append("blast-radius:+1")
    if bumps:
        # Risk signals buy headroom but cannot demote a model Jev already chose
        # for scale or context. The risk ceiling limits escalation, not selection.
        ceiling = max(rank, _index(models, thresholds.risk_ceiling))
        rank = min(rank + bumps, ceiling)
    model = _settle(models, rank, allowed, min_model, max_model)
    if model.key != models[rank].key:
        reasons.append("clamped:%s" % model.key)
    return Decision(model, tuple(reasons), judgment, True)
