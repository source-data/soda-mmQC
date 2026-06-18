"""Tests for collation layout discovery (phase 5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soda_mmqc.core.collation import (
    build_eval_leaf_specs,
    discover_collation_layout,
    validate_manifest_field_patterns,
    validate_manifest_list_alignment,
)
from soda_mmqc.core.eval_manifest import load_eval_manifest

FIXTURES = Path(__file__).parent / "fixtures"
PANEL_SCHEMA = json.loads(
    (FIXTURES / "panel_segmentation_schema.json").read_text()
)
TOY_SCHEMA = json.loads((FIXTURES / "toy_eval_schema.json").read_text())
TOY_GOLD = json.loads((FIXTURES / "toy_eval_gold_baseline.json").read_text())
COLLATED_GOLD = json.loads((FIXTURES / "collated_eval_gold.json").read_text())
COLLATED_PRED = json.loads((FIXTURES / "collated_eval_pred.json").read_text())
TOY_MANIFEST = load_eval_manifest(FIXTURES / "eval_manifest_toy.json")
COLLATED_MANIFEST = load_eval_manifest(FIXTURES / "eval_manifest_collated.json")


class TestDiscoverCollationLayout:
    def test_root_embedding_for_toy(self):
        layout = discover_collation_layout(TOY_SCHEMA, TOY_GOLD, TOY_GOLD)
        assert layout.embedding_prefix == ()
        assert len(layout.predictive_lists) == 1
        assert layout.predictive_lists[0].by_list_key == "panels"

    def test_figures_wrapper_for_collated(self):
        layout = discover_collation_layout(PANEL_SCHEMA, COLLATED_GOLD, COLLATED_PRED)
        assert layout.embedding_prefix == ("figures",)
        assert layout.predictive_lists[0].by_list_key == "figures.panels"
        assert layout.predictive_lists[0].alignment_list_name == "figures[].panels"

        structural = [spec for spec in layout.eval_lists if not spec.is_predictive]
        predictive = [spec for spec in layout.eval_lists if spec.is_predictive]
        assert len(structural) == 1
        assert structural[0].by_list_key == "figures"
        assert len(predictive) == 1

    def test_mismatched_embeddings_raise(self):
        pred = {"figures": [{"panels": []}]}
        gold = {"panels": []}
        with pytest.raises(ValueError, match="different collation embeddings"):
            discover_collation_layout(PANEL_SCHEMA, gold, pred)


class TestManifestValidation:
    def test_collated_manifest_matches_layout(self):
        layout = discover_collation_layout(PANEL_SCHEMA, COLLATED_GOLD, COLLATED_PRED)
        validate_manifest_list_alignment(COLLATED_MANIFEST, PANEL_SCHEMA, layout)

    def test_missing_list_alignment_key_raises(self):
        layout = discover_collation_layout(PANEL_SCHEMA, COLLATED_GOLD, COLLATED_PRED)
        with pytest.raises(ValueError, match="missing keys"):
            validate_manifest_list_alignment(TOY_MANIFEST, PANEL_SCHEMA, layout)


class TestBuildEvalLeafSpecs:
    def test_remapped_schema_leaves_only(self):
        layout = discover_collation_layout(PANEL_SCHEMA, COLLATED_GOLD, COLLATED_PRED)
        specs = build_eval_leaf_specs(PANEL_SCHEMA, layout)
        patterns = {spec.eval_pattern for spec in specs}
        assert "figures[].panels[].label" in patterns
        assert "figures[].panels[].is_micrograph" in patterns
        assert "figures[].figure_label" not in patterns
        assert "panels[].label" not in patterns


class TestValidateManifestFieldPatterns:
    def test_structural_field_rejected(self, tmp_path):
        layout = discover_collation_layout(PANEL_SCHEMA, COLLATED_GOLD, COLLATED_PRED)
        raw = json.loads((FIXTURES / "eval_manifest_collated.json").read_text())
        raw["fields"]["figures[].figure_label"] = raw["fields"]["figures[].panels[].label"]
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")
        manifest = load_eval_manifest(manifest_path)
        with pytest.raises(ValueError, match="structural"):
            validate_manifest_field_patterns(manifest, PANEL_SCHEMA, layout)
