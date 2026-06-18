"""Tests for property-level mean score rollups."""

from __future__ import annotations

from soda_mmqc.core.property_rollup import (
    LAYER1_MEAN_SCORE_ELIGIBLE,
    instance_eligible_for_mean_score,
    property_mean_score,
)


def test_property_mean_score_profiled_excludes_na():
    instances = [
        {"score": 1.0, "layer1": "correct_NA"},
        {"score": 0.0, "layer1": LAYER1_MEAN_SCORE_ELIGIBLE},
    ]
    assert property_mean_score(instances, profiled=True) == 0.0


def test_property_mean_score_profiled_all_na():
    instances = [{"score": 1.0, "layer1": "correct_NA"}]
    assert property_mean_score(instances, profiled=True) == 0.0


def test_property_mean_score_unprofiled_includes_all():
    instances = [
        {"score": 1.0},
        {"score": 0.0},
    ]
    assert property_mean_score(instances, profiled=False) == 0.5


def test_instance_eligible_for_mean_score():
    assert instance_eligible_for_mean_score(
        {"score": 1.0, "layer1": LAYER1_MEAN_SCORE_ELIGIBLE},
        profiled=True,
    )
    assert not instance_eligible_for_mean_score(
        {"score": 1.0, "layer1": "correct_NA"},
        profiled=True,
    )
    assert instance_eligible_for_mean_score({"score": 0.5}, profiled=False)
