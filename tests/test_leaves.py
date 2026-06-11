"""Unit tests for primitive leaf comparison (soda_mmqc.core.leaves)."""

from __future__ import annotations

import math

import pytest
from rapidfuzz import fuzz

from soda_mmqc.core.leaves import (
    LeafComparisonResult,
    StringCompareMode,
    compare_boolean,
    compare_enum_string,
    compare_exact_strings,
    compare_fuzzy_strings,
    compare_number,
    compare_semantic_strings,
    compare_strings,
    fuzzy_ratio,
    score_exact_string,
    score_fuzzy_string,
    score_semantic_string,
)


@pytest.fixture(scope="module")
def sentence_transformer_loaded():
    """Load the default SentenceTransformer once for semantic tests."""
    compare_semantic_strings("warmup", "warmup")
    yield


class TestFuzzyRatio:
    def test_identical(self):
        assert fuzzy_ratio("yes", "yes") == 1.0
        assert fuzzy_ratio("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert fuzzy_ratio("yes", "no") == 0.0
        assert fuzzy_ratio("cat", "dog") == 0.0

    def test_empty_strings(self):
        assert fuzzy_ratio("", "") == 1.0
        assert fuzzy_ratio("", "hello") == 0.0
        assert fuzzy_ratio("hello", "") == 0.0

    def test_typo_partial_match(self):
        assert math.isclose(fuzzy_ratio("hello", "helo"), 88.88888888888889 / 100)

    def test_text_segment(self):
        assert fuzzy_ratio("the cat sat", "the big cat sat") > 0.8


class TestScoreExactString:
    def test_match(self):
        assert score_exact_string("Panel A", "Panel A") == 1.0

    def test_mismatch(self):
        assert score_exact_string("Panel A", "Panel B") == 0.0

    def test_case_sensitive(self):
        assert score_exact_string("Hello", "hello") == 0.0

    def test_empty_strings(self):
        assert score_exact_string("", "") == 1.0


class TestScoreFuzzyString:
    def test_case_insensitive(self):
        assert score_fuzzy_string("YES", "yes") == 1.0

    def test_strips_whitespace(self):
        assert score_fuzzy_string(" yes ", "yes") == 1.0

    def test_partial_overlap(self):
        assert score_fuzzy_string("present", "presentation") > 0.7

    def test_matches_fuzzy_ratio_after_normalize(self):
        pred, exp = "Hello", "helo"
        assert score_fuzzy_string(pred, exp) == fuzzy_ratio(
            pred.lower().strip(), exp.lower().strip()
        )


class TestScoreSemanticString:
    def test_identical_strings(self, sentence_transformer_loaded):
        assert score_semantic_string(
            "The cat is on the mat", "The cat is on the mat"
        ) == 1.0

    def test_synonyms_high_score(self, sentence_transformer_loaded):
        score = score_semantic_string(
            "The cat is on the mat", "The feline is on the mat"
        )
        assert score > 0.7
        assert score <= 1.0

    def test_related_concepts_moderate_score(self, sentence_transformer_loaded):
        score = score_semantic_string(
            "The cat is on the mat", "The dog is on the mat"
        )
        assert score > 0.5
        assert score <= 1.0

    def test_unrelated_low_score(self, sentence_transformer_loaded):
        score = score_semantic_string(
            "The cat is on the mat", "The weather is sunny"
        )
        assert score < 0.5
        assert score >= 0.0

    def test_opposites_moderate_score(self, sentence_transformer_loaded):
        score = score_semantic_string("The room is hot", "The room is cold")
        # Same sentence frame — high similarity despite opposite adjectives
        assert score > 0.5
        assert score <= 1.0

    def test_same_meaning_different_words(self, sentence_transformer_loaded):
        score = score_semantic_string(
            "I love this movie", "I adore this film"
        )
        assert score > 0.7
        assert score <= 1.0


class TestCompareExactStrings:
    def test_match(self):
        result = compare_exact_strings("hello", "hello")
        assert result == LeafComparisonResult(score=1.0)

    def test_mismatch(self):
        result = compare_exact_strings("hello", "world")
        assert result.score == 0.0

    def test_empty_strings(self):
        assert compare_exact_strings("", "").score == 1.0

    def test_one_none(self):
        assert compare_exact_strings(None, "x").score == 0.0
        assert compare_exact_strings("x", None).score == 0.0

    def test_both_none_string_only_schema(self):
        assert compare_exact_strings(None, None).score == 0.0

    def test_both_none_when_allowed(self):
        assert compare_exact_strings(None, None, allows_null=True).score == 1.0

    def test_clamps_score(self):
        result = LeafComparisonResult(score=1.5)
        assert result.score == 1.0


class TestCompareFuzzyStrings:
    def test_synonym_case(self):
        result = compare_fuzzy_strings("Yes", "yes")
        assert result.score == 1.0

    def test_yes_no_zero(self):
        assert compare_fuzzy_strings("yes", "no").score == 0.0

    def test_none_handling(self):
        assert compare_fuzzy_strings(None, None).score == 0.0
        assert compare_fuzzy_strings(None, "yes").score == 0.0

    def test_typo_scores_high(self):
        result = compare_fuzzy_strings("hello", "helo")
        assert result.score == pytest.approx(fuzz.ratio("hello", "helo") / 100.0)


class TestCompareSemanticStrings:
    def test_identical(self, sentence_transformer_loaded):
        result = compare_semantic_strings(
            "The cat is on the mat", "The cat is on the mat"
        )
        assert result.score == 1.0

    def test_unrelated_below_half(self, sentence_transformer_loaded):
        result = compare_semantic_strings(
            "The cat is on the mat", "The weather is sunny"
        )
        assert result.score < 0.5

    def test_synonyms_above_exact_match(self, sentence_transformer_loaded):
        exact = compare_exact_strings(
            "The cat is on the mat", "The feline is on the mat"
        )
        semantic = compare_semantic_strings(
            "The cat is on the mat", "The feline is on the mat"
        )
        assert exact.score == 0.0
        assert semantic.score > 0.5

    def test_none_handling(self, sentence_transformer_loaded):
        assert compare_semantic_strings(None, "yes").score == 0.0
        assert compare_semantic_strings("yes", None).score == 0.0
        assert compare_semantic_strings(None, None).score == 0.0


class TestCompareStringsDispatch:
    def test_exact_mode(self):
        r = compare_strings("a", "a", mode=StringCompareMode.EXACT)
        assert r.score == 1.0

    def test_fuzzy_mode(self):
        r = compare_strings("YES", "yes", mode=StringCompareMode.FUZZY)
        assert r.score == 1.0

    def test_semantic_mode(self, sentence_transformer_loaded):
        r = compare_strings(
            "The cat is on the mat",
            "The cat is on the mat",
            mode=StringCompareMode.SEMANTIC,
        )
        assert r.score == 1.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown string compare mode"):
            compare_strings("a", "b", mode="nope")  # type: ignore[arg-type]


class TestCompareEnumString:
    def test_match(self):
        allowed = ["", "yes", "no"]
        result = compare_enum_string("yes", "yes", allowed)
        assert result.score == 1.0
        assert result.enum_violation is False

    def test_mismatch_allowed_literal(self):
        result = compare_enum_string("no", "yes", ["", "yes", "no"])
        assert result.score == 0.0
        assert result.enum_violation is False

    def test_pred_outside_enum(self):
        result = compare_enum_string("maybe", "yes", ["", "yes", "no"])
        assert result.score == 0.0
        assert result.enum_violation is True

    def test_na_sentinel(self):
        result = compare_enum_string("", "", ["", "yes", "no"])
        assert result.score == 1.0


class TestCompareBoolean:
    def test_match(self):
        assert compare_boolean(True, True).score == 1.0
        assert compare_boolean(False, False).score == 1.0

    def test_mismatch(self):
        assert compare_boolean(True, False).score == 0.0

    def test_none(self):
        assert compare_boolean(None, True).score == 0.0
        assert compare_boolean(True, None).score == 0.0
        assert compare_boolean(None, None).score == 0.0


class TestCompareNumber:
    def test_integer_match(self):
        assert compare_number(1843, 1843).score == 1.0

    def test_integer_mismatch(self):
        assert compare_number(1843, 1900).score == 0.0

    def test_float_match(self):
        assert compare_number(1.5, 1.5).score == 1.0

    def test_int_float_equality(self):
        assert compare_number(1, 1.0).score == 1.0

    def test_none(self):
        assert compare_number(None, 1).score == 0.0
        assert compare_number(1, None).score == 0.0
