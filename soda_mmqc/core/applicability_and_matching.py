"""Applicability (layer 1) and matching (layer 2) labels for leaf instances."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from soda_mmqc.core.eval_manifest import FieldProfile, MatchingMetric


class Layer1Label(str, Enum):
    CORRECT_NA = "correct_NA"
    SPURIOUS_APPLICABLE = "spurious_applicable"
    WITHHELD_APPLICABLE = "withheld_applicable"
    CORRECT_APPLICABLE = "correct_applicable"


class Layer2Label(str, Enum):
    TP = "TP"
    FP = "FP"
    FN = "FN"
    TN = "TN"
    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class LayerReporting:
    """Layer 1 label and optional layer 2 label for one profiled instance."""

    layer1: Layer1Label
    layer2: Optional[Layer2Label] = None


def is_value_present(value: Any) -> bool:
    """True when a JSON value was supplied (``None`` means absent/missing)."""
    return value is not None


def is_applicable(value: Any, na_values: tuple[str, ...]) -> bool:
    """Applicable = present and not an N/A sentinel from the manifest."""
    if not is_value_present(value):
        return False
    if not isinstance(value, str):
        return True
    return value not in na_values


def layer1_label(
    exp_value: Any,
    pred_value: Any,
    profile: FieldProfile,
) -> Layer1Label:
    """Applicability label from gold/pred and ``na_values``."""
    gold_applicable = is_applicable(exp_value, profile.na_values)
    pred_applicable = is_applicable(pred_value, profile.na_values)

    if not gold_applicable and not pred_applicable:
        return Layer1Label.CORRECT_NA
    if not gold_applicable and pred_applicable:
        return Layer1Label.SPURIOUS_APPLICABLE
    if gold_applicable and not pred_applicable:
        return Layer1Label.WITHHELD_APPLICABLE
    return Layer1Label.CORRECT_APPLICABLE


def layer2_label(
    exp_value: Any,
    pred_value: Any,
    profile: FieldProfile,
    score: float,
) -> Layer2Label:
    """Matching label when layer 1 is ``correct_applicable``."""
    if profile.matching_metric == MatchingMetric.BINARY_POLARITY:
        return _binary_polarity_label(exp_value, pred_value, profile)
    if profile.matching_metric == MatchingMetric.MULTICLASS:
        return (
            Layer2Label.MATCH
            if exp_value == pred_value
            else Layer2Label.MISMATCH
        )
    if profile.matching_metric == MatchingMetric.GRADED_STRING:
        threshold = profile.match_threshold
        if threshold is None:
            raise ValueError("graded_string profile requires match_threshold")
        return (
            Layer2Label.MATCH
            if score >= threshold
            else Layer2Label.MISMATCH
        )
    raise ValueError(f"Unknown matching_metric: {profile.matching_metric!r}")


def report_layers(
    exp_value: Any,
    pred_value: Any,
    profile: FieldProfile,
    score: float,
) -> LayerReporting:
    """Compute layer 1 and layer 2 labels for a profiled leaf instance."""
    if not profile.is_profiled:
        raise ValueError("profile must define matching_metric")

    label1 = layer1_label(exp_value, pred_value, profile)
    if label1 is not Layer1Label.CORRECT_APPLICABLE:
        return LayerReporting(layer1=label1, layer2=None)

    label2 = layer2_label(exp_value, pred_value, profile, score)
    return LayerReporting(layer1=label1, layer2=label2)


def _binary_polarity_label(
    exp_value: Any,
    pred_value: Any,
    profile: FieldProfile,
) -> Layer2Label:
    positive = profile.positive_value
    negative = profile.negative_value
    if positive is None or negative is None:
        raise ValueError(
            "binary_polarity requires positive_value and negative_value"
        )

    gold_positive = exp_value == positive
    gold_negative = exp_value == negative
    pred_positive = pred_value == positive
    pred_negative = pred_value == negative

    if gold_positive and pred_positive:
        return Layer2Label.TP
    if gold_negative and pred_negative:
        return Layer2Label.TN
    if gold_negative and pred_positive:
        return Layer2Label.FP
    if gold_positive and pred_negative:
        return Layer2Label.FN

    # Gold-applicable non-polarity enum values: treat non-positive pred as FN,
    # non-negative pred as FP when gold is negative.
    if gold_positive:
        return Layer2Label.FN
    if gold_negative:
        return Layer2Label.FP
    return Layer2Label.MISMATCH
