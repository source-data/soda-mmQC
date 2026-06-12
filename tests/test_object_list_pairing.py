"""Tests for list-of-objects row pairing (phase 4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda_mmqc.core.eval_manifest import FieldProfile, load_eval_manifest
from soda_mmqc.core.leaves import StringCompareMode
from soda_mmqc.core.object_list_pairing import (
    align_object_rows,
    leaf_property_for_alignment_key,
    row_similarity,
    score_alignment_key,
)

FIXTURES = Path(__file__).parent / "fixtures"
TOY_MANIFEST = load_eval_manifest(FIXTURES / "eval_manifest_toy.json")

BASELINE_GOLD_PANELS = [
    {"id": 1, "label": "Fig 1", "status": "yes"},
    {"id": 2, "label": "Fig 2", "status": ""},
]


@pytest.fixture
def panels_label_profile() -> FieldProfile:
    profile = TOY_MANIFEST.profile_for("panels[].label")
    assert profile is not None
    return profile


class TestLeafPropertyForAlignmentKey:
    def test_panels_label(self):
        assert leaf_property_for_alignment_key("panels", "label") == "panels[].label"


class TestScoreAlignmentKey:
    def test_graded_string_exact(self, panels_label_profile: FieldProfile):
        assert (
            score_alignment_key("Fig 1", "Fig 1", panels_label_profile) == 1.0
        )
        assert (
            score_alignment_key("Fig 1", "Fig 2", panels_label_profile) == 0.0
        )

    def test_enum_field_exact(self):
        profile = TOY_MANIFEST.profile_for("panels[].status")
        assert profile is not None
        assert score_alignment_key("yes", "yes", profile) == 1.0
        assert score_alignment_key("yes", "no", profile) == 0.0


class TestRowSimilarity:
    def test_matching_rows(self, panels_label_profile: FieldProfile):
        exp = {"label": "Fig 1", "status": "yes"}
        pred = {"label": "Fig 1", "status": "no"}
        sim = row_similarity(exp, pred, ["label"], (panels_label_profile,))
        assert sim == 1.0

    def test_status_does_not_affect_label_only_similarity(
        self, panels_label_profile: FieldProfile
    ):
        exp = {"label": "Fig 1", "status": "yes"}
        pred = {"label": "Fig 1", "status": "no"}
        assert row_similarity(exp, pred, ["label"], (panels_label_profile,)) == 1.0


class TestAlignObjectRows:
    def test_perfect_match(self):
        result = align_object_rows(
            BASELINE_GOLD_PANELS,
            BASELINE_GOLD_PANELS,
            list_name="panels",
            manifest=TOY_MANIFEST,
        )
        assert result.gold_to_pred == ((0, 0), (1, 1))
        assert result.match_threshold == 1.0
        assert len(result.pair_similarities) == 2

    def test_d2_reordered_rows(self):
        pred = [
            {"id": 9, "label": "Fig 2", "status": ""},
            {"id": 8, "label": "Fig 1", "status": "no"},
        ]
        result = align_object_rows(
            BASELINE_GOLD_PANELS,
            pred,
            list_name="panels",
            manifest=TOY_MANIFEST,
        )
        assert result.gold_to_pred == ((0, 1), (1, 0))
        assert result.pred_index_for_gold(0) == 1
        assert result.pred_index_for_gold(1) == 0

    def test_d4_missing_panel(self):
        pred = [{"id": 1, "label": "Fig 1", "status": "yes"}]
        result = align_object_rows(
            BASELINE_GOLD_PANELS,
            pred,
            list_name="panels",
            manifest=TOY_MANIFEST,
        )
        assert result.gold_to_pred == ((0, 0),)
        assert result.pred_index_for_gold(1) is None

    def test_d5_extra_panel(self):
        pred = BASELINE_GOLD_PANELS + [
            {"id": 99, "label": "Fig 99", "status": "no"},
        ]
        result = align_object_rows(
            BASELINE_GOLD_PANELS,
            pred,
            list_name="panels",
            manifest=TOY_MANIFEST,
        )
        assert result.gold_to_pred == ((0, 0), (1, 1))
        assert len(result.pair_similarities) == 2
        assert all(pred != 2 for _, pred, _ in result.pair_similarities)

    def test_missing_list_alignment_raises(self):
        with pytest.raises(ValueError, match="list_alignment"):
            align_object_rows([], [], list_name="unknown", manifest=TOY_MANIFEST)

    def test_both_empty(self):
        result = align_object_rows([], [], list_name="panels", manifest=TOY_MANIFEST)
        assert result.gold_to_pred == ()
        assert result.pair_similarities == ()


class TestSemanticAlignmentKey:
    def test_uses_string_compare_from_fields_profile(self):
        manifest = load_eval_manifest(FIXTURES / "eval_manifest_toy.json")
        profile = manifest.profile_for("outputs[].from_the_caption")
        assert profile is not None
        assert profile.string_compare == StringCompareMode.SEMANTIC

        def embed(texts):
            import torch

            return torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        from soda_mmqc.core.leaves import compare_strings

        score = compare_strings(
            "caption text",
            "caption text",
            mode=StringCompareMode.SEMANTIC,
            embedder=embed,
        ).score
        assert score == 1.0

        exp_row = {"from_the_caption": "caption text"}
        pred_row = {"from_the_caption": "caption text"}
        sim = row_similarity(
            exp_row,
            pred_row,
            ["from_the_caption"],
            (profile,),
            embedder=embed,
        )
        assert sim == 1.0
