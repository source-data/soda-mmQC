"""List alignment for the flat evaluation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from soda_mmqc.core.eval_manifest import EvalManifest, FieldProfile, MatchingMetric
from soda_mmqc.core.leaves import (
    StringCompareMode,
    _clamp_score,
    compare_strings,
)

ElementSimilarity = Callable[[Any, Any], float]


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """TP/FP/FN counts from Hungarian matching with a threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    pairs: tuple[tuple[int, int, float], ...]


@dataclass(frozen=True)
class PrimitiveListAlignmentResult:
    """Alignment outcome for one list-of-primitives leaf instance."""

    score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    pairs: tuple[tuple[int, int, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _clamp_score(self.score))


@dataclass(frozen=True)
class ObjectListAlignmentResult:
    """Row alignment for a list of objects (phase 4b)."""

    gold_to_pred: tuple[tuple[int, int], ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    pair_similarities: tuple[tuple[int, int, float], ...]

    def pred_index_for_gold(self, gold_index: int) -> Optional[int]:
        for gold, pred in self.gold_to_pred:
            if gold == gold_index:
                return pred
        return None


def exact_primitive_similarity(exp: Any, pred: Any) -> float:
    """Exact equality similarity for JSON primitive list elements."""
    if exp is None or pred is None:
        return 0.0
    return 1.0 if exp == pred else 0.0


def align_primitive_lists(
    exp: Sequence[Any],
    pred: Sequence[Any],
    *,
    similarity: ElementSimilarity = exact_primitive_similarity,
    match_threshold: float = 1.0,
) -> PrimitiveListAlignmentResult:
    """Align two primitive arrays with Hungarian matching.

    Aggregate **score** = mean over ``n_all = max(len(exp), len(pred))``,
    counting unmatched slots and pairs below ``match_threshold`` as 0.
    """
    if not exp and not pred:
        return PrimitiveListAlignmentResult(
            score=1.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            pairs=(),
        )

    n_exp = len(exp)
    n_pred = len(pred)
    n_all = max(n_exp, n_pred)
    if n_all == 0:
        return PrimitiveListAlignmentResult(
            score=1.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            pairs=(),
        )

    matrix = build_similarity_matrix(exp, pred, similarity)
    diagnostics = _alignment_diagnostics(matrix, n_exp, n_pred, match_threshold)
    score_sum = sum(sim for _, _, sim in diagnostics.pairs if sim >= match_threshold)
    mean_score = score_sum / n_all if n_all else 1.0

    return PrimitiveListAlignmentResult(
        score=mean_score,
        true_positives=diagnostics.true_positives,
        false_positives=diagnostics.false_positives,
        false_negatives=diagnostics.false_negatives,
        pairs=diagnostics.pairs,
    )


def align_object_lists(
    exp_rows: Sequence[Mapping[str, Any]],
    pred_rows: Sequence[Mapping[str, Any]],
    *,
    list_name: str,
    manifest: EvalManifest,
    embedder: Optional[Any] = None,
) -> ObjectListAlignmentResult:
    """Align object list rows using manifest ``list_alignment`` keys.

    Row similarity ``s(i,j)`` is the mean leaf score over alignment keys.
    Each key uses the ``fields`` profile for ``{list_name}[].{key}``
    (``string_compare``, ``match_threshold`` for graded strings; exact
    match for enum / polarity fields). Pairs with ``s(i,j)`` below the
    mean key threshold are not returned in ``gold_to_pred``.
    """
    alignment_keys = manifest.alignment_keys_for(list_name)
    if not alignment_keys:
        raise ValueError(
            f"list_alignment[{list_name!r}] is required for object list alignment"
        )

    key_profiles = _alignment_key_profiles(list_name, alignment_keys, manifest)
    match_threshold = _row_match_threshold(key_profiles)

    n_exp = len(exp_rows)
    n_pred = len(pred_rows)
    if n_exp == 0 and n_pred == 0:
        return ObjectListAlignmentResult(
            gold_to_pred=(),
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            pair_similarities=(),
        )

    matrix = build_object_similarity_matrix(
        exp_rows,
        pred_rows,
        alignment_keys,
        key_profiles,
        embedder=embedder,
    )
    diagnostics = _alignment_diagnostics(matrix, n_exp, n_pred, match_threshold)

    gold_to_pred = tuple(
        (gold, pred)
        for gold, pred, sim in diagnostics.pairs
        if sim >= match_threshold
    )
    return ObjectListAlignmentResult(
        gold_to_pred=gold_to_pred,
        true_positives=diagnostics.true_positives,
        false_positives=diagnostics.false_positives,
        false_negatives=diagnostics.false_negatives,
        pair_similarities=diagnostics.pairs,
    )


def build_similarity_matrix(
    exp: Sequence[Any],
    pred: Sequence[Any],
    similarity: ElementSimilarity,
) -> np.ndarray:
    """All-pairs element similarity matrix ``[len(exp), len(pred)]``."""
    if not exp or not pred:
        return np.zeros((len(exp), len(pred)), dtype=float)
    return np.array(
        [
            [similarity(exp[i], pred[j]) for j in range(len(pred))]
            for i in range(len(exp))
        ],
        dtype=float,
    )


def build_object_similarity_matrix(
    exp_rows: Sequence[Mapping[str, Any]],
    pred_rows: Sequence[Mapping[str, Any]],
    alignment_keys: Sequence[str],
    key_profiles: Sequence[FieldProfile],
    *,
    embedder: Optional[Any] = None,
) -> np.ndarray:
    """Row similarity matrix for list-of-objects alignment."""
    if not exp_rows or not pred_rows:
        return np.zeros((len(exp_rows), len(pred_rows)), dtype=float)
    return np.array(
        [
            [
                row_similarity(
                    exp_rows[i],
                    pred_rows[j],
                    alignment_keys,
                    key_profiles,
                    embedder=embedder,
                )
                for j in range(len(pred_rows))
            ]
            for i in range(len(exp_rows))
        ],
        dtype=float,
    )


def row_similarity(
    exp_row: Mapping[str, Any],
    pred_row: Mapping[str, Any],
    alignment_keys: Sequence[str],
    key_profiles: Sequence[FieldProfile],
    *,
    embedder: Optional[Any] = None,
) -> float:
    """Mean alignment-key score between two object rows."""
    if not alignment_keys:
        return 0.0
    scores = [
        score_alignment_key(
            exp_row.get(key),
            pred_row.get(key),
            profile,
            embedder=embedder,
        )
        for key, profile in zip(alignment_keys, key_profiles)
    ]
    return sum(scores) / len(scores)


def score_alignment_key(
    exp_value: Any,
    pred_value: Any,
    profile: FieldProfile,
    *,
    embedder: Optional[Any] = None,
) -> float:
    """Leaf score for one alignment key using its manifest field profile."""
    if profile.matching_metric == MatchingMetric.GRADED_STRING:
        if profile.string_compare is None:
            raise ValueError("graded_string alignment key requires string_compare")
        if not isinstance(exp_value, str) or not isinstance(pred_value, str):
            return exact_primitive_similarity(exp_value, pred_value)
        return compare_strings(
            pred_value,
            exp_value,
            mode=profile.string_compare,
            embedder=embedder,
        ).score

    if isinstance(exp_value, str) and isinstance(pred_value, str):
        return compare_strings(
            pred_value,
            exp_value,
            mode=StringCompareMode.EXACT,
        ).score

    return exact_primitive_similarity(exp_value, pred_value)


def leaf_property_for_alignment_key(list_name: str, field_name: str) -> str:
    """Manifest path pattern for a row alignment field."""
    return f"{list_name}[].{field_name}"


def _alignment_key_profiles(
    list_name: str,
    alignment_keys: Sequence[str],
    manifest: EvalManifest,
) -> tuple[FieldProfile, ...]:
    profiles: list[FieldProfile] = []
    for key in alignment_keys:
        path = leaf_property_for_alignment_key(list_name, key)
        profile = manifest.profile_for(path)
        if profile is None:
            profiles.append(FieldProfile())
        else:
            profiles.append(profile)
    return tuple(profiles)


def _row_match_threshold(key_profiles: Sequence[FieldProfile]) -> float:
    thresholds: list[float] = []
    for profile in key_profiles:
        if profile.matching_metric == MatchingMetric.GRADED_STRING:
            if profile.match_threshold is None:
                raise ValueError("graded_string alignment key requires match_threshold")
            thresholds.append(profile.match_threshold)
        else:
            thresholds.append(1.0)
    if not thresholds:
        return 1.0
    return sum(thresholds) / len(thresholds)


def _alignment_diagnostics(
    matrix: np.ndarray,
    n_exp: int,
    n_pred: int,
    match_threshold: float,
) -> AlignmentDiagnostics:
    row_indices, col_indices = _hungarian_assignment(matrix)

    matched_pairs: list[tuple[int, int, float]] = []
    matched_exp: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0

    for row, col in zip(row_indices, col_indices):
        sim = float(matrix[row, col])
        matched_pairs.append((int(row), int(col), sim))
        matched_exp.add(int(row))
        matched_pred.add(int(col))
        if sim >= match_threshold:
            tp += 1

    fn = (n_exp - len(matched_exp)) + sum(
        1
        for row, _, sim in matched_pairs
        if row < n_exp and sim < match_threshold
    )
    fp = (n_pred - len(matched_pred)) + sum(
        1
        for _, col, sim in matched_pairs
        if col < n_pred and sim < match_threshold
    )

    return AlignmentDiagnostics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        pairs=tuple(matched_pairs),
    )


def _hungarian_assignment(
    similarity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-weight bipartite matching via ``linear_sum_assignment``."""
    if similarity_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cost = -similarity_matrix
    return linear_sum_assignment(cost)
