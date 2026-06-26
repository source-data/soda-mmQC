"""Shared rollups for ``by_property`` summaries."""

from __future__ import annotations

from typing import Any, Sequence

LAYER1_MEAN_SCORE_ELIGIBLE = "correct_applicable"


def _instance_score(instance: Any) -> float | None:
    score = instance.get("score") if isinstance(instance, dict) else instance.score
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _instance_layer1(instance: Any) -> str | None:
    layer1 = instance.get("layer1") if isinstance(instance, dict) else instance.layer1
    return layer1 if isinstance(layer1, str) else None


def property_mean_score(
    instances: Sequence[Any],
    *,
    profiled: bool,
) -> float:
    """Mean instance score for one leaf property (Layer 2 rollup when profiled)."""
    scores: list[float] = []
    for instance in instances:
        score = _instance_score(instance)
        if score is None:
            continue
        if profiled and _instance_layer1(instance) != LAYER1_MEAN_SCORE_ELIGIBLE:
            continue
        scores.append(score)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def instance_eligible_for_mean_score(
    instance: dict[str, Any],
    *,
    profiled: bool,
) -> bool:
    """Whether an instance contributes to ``mean_score`` and score scatter plots."""
    if _instance_score(instance) is None:
        return False
    if profiled:
        return instance.get("layer1") == LAYER1_MEAN_SCORE_ELIGIBLE
    return True
