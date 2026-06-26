"""Shared bipartite matching for list comparison.

Domain-agnostic utilities: similarity matrix, Hungarian assignment, and
threshold-gated pair lists. Used by extended-primitive leaf compare and
object-list row pairing.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

ElementSimilarity = Callable[[Any, Any], float]

MatchedPair = tuple[int, int, float]


def build_similarity_matrix(
    left: Sequence[Any],
    right: Sequence[Any],
    similarity: ElementSimilarity,
) -> np.ndarray:
    """All-pairs similarity matrix ``[len(left), len(right)]``."""
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=float)
    return np.array(
        [
            [similarity(left[i], right[j]) for j in range(len(right))]
            for i in range(len(left))
        ],
        dtype=float,
    )


def hungarian_assignment(
    similarity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-weight bipartite matching via ``linear_sum_assignment``."""
    if similarity_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cost = -similarity_matrix
    return linear_sum_assignment(cost)


def hungarian_match_pairs(similarity_matrix: np.ndarray) -> tuple[MatchedPair, ...]:
    """Hungarian assignment with similarity for each matched pair."""
    row_indices, col_indices = hungarian_assignment(similarity_matrix)
    return tuple(
        (int(row), int(col), float(similarity_matrix[row, col]))
        for row, col in zip(row_indices, col_indices)
    )


def pairs_at_threshold(
    pairs: Sequence[MatchedPair],
    threshold: float,
) -> tuple[MatchedPair, ...]:
    """Pairs whose similarity is at or above ``threshold``."""
    return tuple(pair for pair in pairs if pair[2] >= threshold)


def mean_gated_similarity(
    pairs: Sequence[MatchedPair],
    *,
    n_left: int,
    n_right: int,
    threshold: float,
) -> float:
    """Mean similarity over ``max(n_left, n_right)`` slots.

    Pairs below ``threshold`` and unmatched slots count as zero contribution.
    Both sides empty → ``1.0``.
    """
    n_all = max(n_left, n_right)
    if n_all == 0:
        return 1.0
    score_sum = sum(sim for _, _, sim in pairs if sim >= threshold)
    return score_sum / n_all
