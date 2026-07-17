"""Tests for applicability (layer 1) and matching (layer 2) reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda_mmqc.core.applicability_and_matching import (
    Layer1Label,
    Layer2Label,
    is_applicable,
    is_value_present,
    layer1_label,
    layer2_label,
    report_layers,
)
from soda_mmqc.core.eval_manifest import FieldProfile, MatchingMetric, load_eval_manifest

FIXTURES = Path(__file__).parent / "fixtures"
TOY_MANIFEST = load_eval_manifest(FIXTURES / "eval_manifest_toy.json")


@pytest.fixture
def item_status_profile() -> FieldProfile:
    profile = TOY_MANIFEST.profile_for("item.status")
    assert profile is not None
    return profile


@pytest.fixture
def item_label_profile() -> FieldProfile:
    profile = TOY_MANIFEST.profile_for("item.label")
    assert profile is not None
    return profile


@pytest.fixture
def panels_status_profile() -> FieldProfile:
    profile = TOY_MANIFEST.profile_for("panels[].status")
    assert profile is not None
    return profile


class TestPresenceAndApplicability:
    def test_is_value_present(self):
        assert is_value_present("") is True
        assert is_value_present(0) is True
        assert is_value_present(False) is True
        assert is_value_present(None) is False

    def test_is_applicable_with_na_sentinel(self):
        assert is_applicable("", ("",)) is False
        assert is_applicable("yes", ("",)) is True
        assert is_applicable(None, ("",)) is False

    def test_non_string_always_applicable_when_present(self):
        assert is_applicable(1843, ()) is True
        assert is_applicable(None, ()) is False


class TestLayer1:
    def test_correct_na(self, item_status_profile: FieldProfile):
        assert (
            layer1_label("", "", item_status_profile)
            == Layer1Label.CORRECT_NA
        )

    def test_spurious_applicable(self, item_status_profile: FieldProfile):
        assert (
            layer1_label("", "yes", item_status_profile)
            == Layer1Label.SPURIOUS_APPLICABLE
        )

    def test_withheld_applicable(self, item_label_profile: FieldProfile):
        assert (
            layer1_label("Panel A", None, item_label_profile)
            == Layer1Label.WITHHELD_APPLICABLE
        )

    def test_correct_applicable(self, item_label_profile: FieldProfile):
        assert (
            layer1_label("Panel A", "Panel B", item_label_profile)
            == Layer1Label.CORRECT_APPLICABLE
        )


class TestLayer2BinaryPolarity:
    def test_tp(self, panels_status_profile: FieldProfile):
        assert (
            layer2_label("yes", "yes", panels_status_profile, 1.0)
            == Layer2Label.TP
        )

    def test_tn(self, panels_status_profile: FieldProfile):
        assert (
            layer2_label("no", "no", panels_status_profile, 1.0)
            == Layer2Label.TN
        )

    def test_fp(self, panels_status_profile: FieldProfile):
        assert (
            layer2_label("no", "yes", panels_status_profile, 0.0)
            == Layer2Label.FP
        )

    def test_fn(self, panels_status_profile: FieldProfile):
        assert (
            layer2_label("yes", "no", panels_status_profile, 0.0)
            == Layer2Label.FN
        )

    def test_invalid_gold_value_is_mismatch(self, panels_status_profile: FieldProfile):
        assert (
            layer2_label("[]", "no", panels_status_profile, 0.0)
            == Layer2Label.MISMATCH
        )

class TestLayer2GradedString:
    def test_match_at_threshold(self, item_label_profile: FieldProfile):
        assert (
            layer2_label("Panel A", "Panel A", item_label_profile, 1.0)
            == Layer2Label.MATCH
        )

    def test_mismatch_below_threshold(self, item_label_profile: FieldProfile):
        assert (
            layer2_label("Panel A", "Panel B", item_label_profile, 0.0)
            == Layer2Label.MISMATCH
        )

    def test_match_at_partial_score_above_threshold(self):
        profile = FieldProfile(
            matching_metric=MatchingMetric.GRADED_STRING,
            match_threshold=0.8,
        )
        assert (
            layer2_label("a", "b", profile, 0.85) == Layer2Label.MATCH
        )

    def test_requires_match_threshold(self):
        profile = FieldProfile(matching_metric=MatchingMetric.GRADED_STRING)
        with pytest.raises(ValueError, match="match_threshold"):
            layer2_label("a", "b", profile, 0.9)


class TestLayer2Multiclass:
    def test_match_and_mismatch(self):
        profile = FieldProfile(matching_metric=MatchingMetric.MULTICLASS)
        assert layer2_label("a", "a", profile, 1.0) == Layer2Label.MATCH
        assert layer2_label("a", "b", profile, 0.0) == Layer2Label.MISMATCH


class TestReportLayers:
    def test_skips_layer2_when_not_correct_applicable(
        self, item_status_profile: FieldProfile
    ):
        result = report_layers("", "yes", item_status_profile, 0.0)
        assert result.layer1 == Layer1Label.SPURIOUS_APPLICABLE
        assert result.layer2 is None

    def test_reports_both_layers(self, item_label_profile: FieldProfile):
        result = report_layers("Panel A", "Panel B", item_label_profile, 0.0)
        assert result.layer1 == Layer1Label.CORRECT_APPLICABLE
        assert result.layer2 == Layer2Label.MISMATCH

    def test_requires_profile(self):
        with pytest.raises(ValueError, match="matching_metric"):
            report_layers("a", "b", FieldProfile(), 1.0)


class TestToyExamplesC:
    """Mirror evaluation-toy-examples.md section C."""

    def test_c1_wrong_item_label(self, item_label_profile: FieldProfile):
        result = report_layers("Panel A", "Panel B", item_label_profile, 0.0)
        assert result.layer1 == Layer1Label.CORRECT_APPLICABLE
        assert result.layer2 == Layer2Label.MISMATCH

    def test_c2_spurious_status(self, item_status_profile: FieldProfile):
        result = report_layers("", "yes", item_status_profile, 0.0)
        assert result.layer1 == Layer1Label.SPURIOUS_APPLICABLE
        assert result.layer2 is None

    def test_c3_missing_item_label(self, item_label_profile: FieldProfile):
        result = report_layers("Panel A", None, item_label_profile, 0.0)
        assert result.layer1 == Layer1Label.WITHHELD_APPLICABLE
        assert result.layer2 is None


class TestToyExamplesAAndD:
    def test_a_perfect_item_label(self, item_label_profile: FieldProfile):
        result = report_layers("Panel A", "Panel A", item_label_profile, 1.0)
        assert result.layer2 == Layer2Label.MATCH

    def test_a_correct_na_status(self, item_status_profile: FieldProfile):
        result = report_layers("", "", item_status_profile, 1.0)
        assert result.layer1 == Layer1Label.CORRECT_NA
        assert result.layer2 is None

    def test_a_panel_status_tp(self, panels_status_profile: FieldProfile):
        result = report_layers("yes", "yes", panels_status_profile, 1.0)
        assert result.layer1 == Layer1Label.CORRECT_APPLICABLE
        assert result.layer2 == Layer2Label.TP

    def test_d2_status_fn(self, panels_status_profile: FieldProfile):
        result = report_layers("yes", "no", panels_status_profile, 0.0)
        assert result.layer1 == Layer1Label.CORRECT_APPLICABLE
        assert result.layer2 == Layer2Label.FN

    def test_d3_spurious_panel_status(self, panels_status_profile: FieldProfile):
        result = report_layers("", "yes", panels_status_profile, 0.0)
        assert result.layer1 == Layer1Label.SPURIOUS_APPLICABLE

    def test_d4_missing_pred_on_na_gold(self, panels_status_profile: FieldProfile):
        result = report_layers("", None, panels_status_profile, 0.0)
        assert result.layer1 == Layer1Label.CORRECT_NA
        assert result.layer2 is None
