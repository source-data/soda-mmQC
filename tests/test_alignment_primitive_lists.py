"""Tests for list-of-primitives alignment (phase 4a)."""

from __future__ import annotations

import pytest

from soda_mmqc.core.alignment import (
    align_primitive_lists,
    build_similarity_matrix,
    exact_primitive_similarity,
)


class TestExactPrimitiveSimilarity:
    def test_match(self):
        assert exact_primitive_similarity("alpha", "alpha") == 1.0
        assert exact_primitive_similarity(1, 1) == 1.0

    def test_mismatch(self):
        assert exact_primitive_similarity("alpha", "beta") == 0.0

    def test_none(self):
        assert exact_primitive_similarity(None, "x") == 0.0


class TestBuildSimilarityMatrix:
    def test_shape_and_values(self):
        matrix = build_similarity_matrix(
            ["alpha", "beta"],
            ["alpha", "gamma"],
            exact_primitive_similarity,
        )
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0


class TestAlignPrimitiveLists:
    def test_both_empty(self):
        result = align_primitive_lists([], [])
        assert result.score == 1.0
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_perfect_match(self):
        tags = ["alpha", "beta"]
        result = align_primitive_lists(tags, tags)
        assert result.score == 1.0
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_b1_extra_element(self):
        """Toy example B.1 — extra pred element."""
        exp = ["alpha", "beta"]
        pred = ["alpha", "beta", "gamma"]
        result = align_primitive_lists(exp, pred)
        assert result.score == pytest.approx(2 / 3)
        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_b2_substituted_element(self):
        """Toy example B.2 — one wrong element."""
        exp = ["alpha", "beta", "gamma"]
        pred = ["alpha", "beta", "delta"]
        result = align_primitive_lists(exp, pred)
        assert result.score == pytest.approx(2 / 3)
        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_b3_total_mismatch(self):
        """Toy example B.3 — single mismatched element."""
        result = align_primitive_lists(["alpha"], ["omega"])
        assert result.score == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_missing_pred_all_fn(self):
        result = align_primitive_lists(["alpha", "beta"], [])
        assert result.score == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 0
        assert result.false_negatives == 2

    def test_missing_gold_all_fp(self):
        result = align_primitive_lists([], ["alpha"])
        assert result.score == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_fuzzy_similarity_with_lower_threshold(self):
        def fuzzy(a: str, b: str) -> float:
            return 1.0 if a.lower() == b.lower() else 0.0

        result = align_primitive_lists(
            ["Yes"],
            ["yes"],
            similarity=fuzzy,
            match_threshold=1.0,
        )
        assert result.score == 1.0
        assert result.true_positives == 1

    def test_pairs_record_similarity(self):
        result = align_primitive_lists(["alpha"], ["alpha"])
        assert result.pairs == ((0, 0, 1.0),)

    def test_score_clamped(self):
        result = align_primitive_lists(["a"], ["a"])
        assert 0.0 <= result.score <= 1.0
