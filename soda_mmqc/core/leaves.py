"""Primitive leaf comparison for the flat evaluation model.

Each function compares two terminal JSON values and returns a score in [0, 1].
Layer 1 / layer 2 reporting and match_threshold are applied by the outer
comparator (see thinking/evaluation-scoring.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from rapidfuzz import fuzz

from soda_mmqc.config import DEFAULT_SENTENCE_TRANSFORMER_MODEL


class StringCompareMode(str, Enum):
    """Manifest ``string_compare`` values."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class LeafComparisonResult:
    """Result of comparing two primitive leaf values."""

    score: float
    enum_violation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _clamp_score(self.score))


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _leaf(score: float, *, enum_violation: bool = False) -> LeafComparisonResult:
    return LeafComparisonResult(score=score, enum_violation=enum_violation)


def fuzzy_ratio(s1: str, s2: str) -> float:
    """Normalized edit-distance ratio via rapidfuzz, in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def score_exact_string(pred: str, exp: str) -> float:
    """Character-level equality."""
    return 1.0 if pred == exp else 0.0


def score_fuzzy_string(pred: str, exp: str) -> float:
    """rapidfuzz ``fuzz.ratio`` after lowercase + strip."""
    return fuzzy_ratio(pred.lower().strip(), exp.lower().strip())


def score_semantic_string(
    pred: str,
    exp: str,
    *,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    embedder: Optional[Callable[[Sequence[str]], Any]] = None,
) -> float:
    """Cosine similarity of sentence embeddings, mapped to [0, 1]."""
    if embedder is None:
        embedder = _default_semantic_embedder(model_name)
    vectors = embedder([pred, exp])
    return _cosine_similarity_01(vectors[0], vectors[1])


def compare_exact_strings(
    pred: Optional[str],
    exp: Optional[str],
    *,
    allows_null: bool = False,
) -> LeafComparisonResult:
    """Compare two strings with exact equality."""
    early = _resolve_optional_string_pair(pred, exp, allows_null=allows_null)
    if early is not None:
        return _leaf(early)
    assert pred is not None and exp is not None
    return _leaf(score_exact_string(pred, exp))


def compare_fuzzy_strings(
    pred: Optional[str],
    exp: Optional[str],
    *,
    allows_null: bool = False,
) -> LeafComparisonResult:
    """Compare two strings with rapidfuzz edit-distance ratio."""
    early = _resolve_optional_string_pair(pred, exp, allows_null=allows_null)
    if early is not None:
        return _leaf(early)
    assert pred is not None and exp is not None
    return _leaf(score_fuzzy_string(pred, exp))


def compare_semantic_strings(
    pred: Optional[str],
    exp: Optional[str],
    *,
    allows_null: bool = False,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    embedder: Optional[Callable[[Sequence[str]], Any]] = None,
) -> LeafComparisonResult:
    """Compare two strings with embedding cosine similarity."""
    early = _resolve_optional_string_pair(pred, exp, allows_null=allows_null)
    if early is not None:
        return _leaf(early)
    assert pred is not None and exp is not None
    return _leaf(
        score_semantic_string(
            pred, exp, model_name=model_name, embedder=embedder
        )
    )


def compare_strings(
    pred: Optional[str],
    exp: Optional[str],
    *,
    mode: StringCompareMode = StringCompareMode.EXACT,
    allows_null: bool = False,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    embedder: Optional[Callable[[Sequence[str]], Any]] = None,
) -> LeafComparisonResult:
    """Dispatch string comparison by manifest ``string_compare`` mode."""
    if mode == StringCompareMode.EXACT:
        return compare_exact_strings(pred, exp, allows_null=allows_null)
    if mode == StringCompareMode.FUZZY:
        return compare_fuzzy_strings(pred, exp, allows_null=allows_null)
    if mode == StringCompareMode.SEMANTIC:
        return compare_semantic_strings(
            pred,
            exp,
            allows_null=allows_null,
            model_name=model_name,
            embedder=embedder,
        )
    raise ValueError(f"Unknown string compare mode: {mode!r}")


def compare_enum_string(
    pred: Optional[str],
    exp: Optional[str],
    allowed: Sequence[str],
) -> LeafComparisonResult:
    """Schema enum string: pred must be an allowed literal; then exact match."""
    if pred is not None and pred not in allowed:
        return _leaf(0.0, enum_violation=True)
    return compare_exact_strings(pred, exp)


def compare_boolean(
    pred: Optional[bool],
    exp: Optional[bool],
) -> LeafComparisonResult:
    """Exact boolean equality."""
    return _leaf(_score_exact_value(pred, exp))


def compare_number(
    pred: Optional[int | float],
    exp: Optional[int | float],
) -> LeafComparisonResult:
    """Exact numeric equality (JSON-parsed int or float)."""
    return _leaf(_score_exact_value(pred, exp))


def _score_exact_value(pred: Any, exp: Any) -> float:
    if pred is None or exp is None:
        return 0.0
    return 1.0 if pred == exp else 0.0


def _resolve_optional_string_pair(
    pred: Optional[str],
    exp: Optional[str],
    *,
    allows_null: bool,
) -> Optional[float]:
    """Return a fixed score when null/absent rules apply, else None to continue."""
    if pred is None and exp is None:
        return 1.0 if allows_null else 0.0
    if pred is None or exp is None:
        return 0.0
    return None


_sentence_transformer_cache: dict[str, Any] = {}


def _get_sentence_transformer(model_name: str) -> Any:
    if model_name not in _sentence_transformer_cache:
        from sentence_transformers import SentenceTransformer

        from soda_mmqc.config import get_device

        _sentence_transformer_cache[model_name] = SentenceTransformer(
            model_name, device=get_device()
        )
    return _sentence_transformer_cache[model_name]


def _default_semantic_embedder(
    model_name: str,
) -> Callable[[Sequence[str]], Any]:
    """Lazy-load SentenceTransformer; cache per model name."""

    def _encode(texts: Sequence[str]) -> Any:
        model = _get_sentence_transformer(model_name)
        return model.encode(list(texts), convert_to_tensor=True)

    return _encode


def _cosine_similarity_01(vec_a: Any, vec_b: Any) -> float:
    import torch

    similarity = torch.nn.functional.cosine_similarity(
        vec_a.unsqueeze(0),
        vec_b.unsqueeze(0),
    )
    raw = float(similarity.item())
    return _clamp_score((raw + 1.0) / 2.0)
