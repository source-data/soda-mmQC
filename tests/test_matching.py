"""Tests for shared bipartite matching (phase 4.1)."""

from __future__ import annotations

import numpy as np
import pytest

from soda_mmqc.core.matching import (
    build_similarity_matrix,
    hungarian_assignment,
    hungarian_match_pairs,
    mean_gated_similarity,
    pairs_at_threshold,
)


def _exact(a: str, b: str) -> float:
    return 1.0 if a == b else 0.0


class TestBuildSimilarityMatrix:
    def test_shape_and_values(self):
        matrix = build_similarity_matrix(
            ["alpha", "beta"],
            ["alpha", "gamma"],
            _exact,
        )
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0

    def test_empty_left(self):
        matrix = build_similarity_matrix([], ["a"], _exact)
        assert matrix.shape == (0, 1)

    def test_empty_right(self):
        matrix = build_similarity_matrix(["a"], [], _exact)
        assert matrix.shape == (1, 0)


class TestHungarianAssignment:
    def test_empty_matrix(self):
        rows, cols = hungarian_assignment(np.zeros((0, 0)))
        assert len(rows) == 0
        assert len(cols) == 0

    def test_maximizes_weight(self):
        matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        rows, cols = hungarian_assignment(matrix)
        pairs = list(zip(rows.tolist(), cols.tolist()))
        assert (0, 0) in pairs
        assert (1, 1) in pairs


class TestHungarianMatchPairs:
    def test_returns_similarities(self):
        matrix = build_similarity_matrix(["alpha"], ["alpha"], _exact)
        pairs = hungarian_match_pairs(matrix)
        assert pairs == ((0, 0, 1.0),)

    def test_reordered_optimal(self):
        matrix = build_similarity_matrix(
            ["a", "b"],
            ["b", "a"],
            _exact,
        )
        pairs = hungarian_match_pairs(matrix)
        assert set((g, p) for g, p, _ in pairs) == {(0, 1), (1, 0)}


class TestPairsAtThreshold:
    def test_filters_below_threshold(self):
        pairs = ((0, 0, 1.0), (1, 1, 0.5))
        assert pairs_at_threshold(pairs, 1.0) == ((0, 0, 1.0),)
        assert pairs_at_threshold(pairs, 0.5) == pairs


class TestMeanGatedSimilarity:
    def test_both_empty(self):
        assert mean_gated_similarity((), n_left=0, n_right=0, threshold=1.0) == 1.0

    def test_perfect_match(self):
        pairs = ((0, 0, 1.0), (1, 1, 1.0))
        assert mean_gated_similarity(
            pairs, n_left=2, n_right=2, threshold=1.0
        ) == pytest.approx(1.0)

    def test_extra_pred_slot(self):
        """Toy B.1 style: two matches, one extra pred slot."""
        pairs = ((0, 0, 1.0), (1, 1, 1.0))
        assert mean_gated_similarity(
            pairs, n_left=2, n_right=3, threshold=1.0
        ) == pytest.approx(2 / 3)

    def test_below_threshold_counts_as_zero(self):
        pairs = ((0, 0, 0.5),)
        assert mean_gated_similarity(
            pairs, n_left=1, n_right=1, threshold=1.0
        ) == pytest.approx(0.0)
