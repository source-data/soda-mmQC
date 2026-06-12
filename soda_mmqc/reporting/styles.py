"""Outcome orders and colour maps for flat evaluation reporting."""

from __future__ import annotations

LAYER_S_ORDER = ("correct_row", "missing_row", "spurious_row")

LAYER1_ORDER = (
    "correct_NA",
    "correct_applicable",
    "withheld_applicable",
    "spurious_applicable",
)

LAYER2_BINARY_ORDER = ("TP", "TN", "FP", "FN")
LAYER2_GRADED_ORDER = ("match", "mismatch")

LAYER_S_COLORS = {
    "correct_row": "#16a34a",
    "missing_row": "#ca8a04",
    "spurious_row": "#dc2626",
}

LAYER1_COLORS = {
    "correct_NA": "#94a3b8",
    "correct_applicable": "#16a34a",
    "withheld_applicable": "#ca8a04",
    "spurious_applicable": "#dc2626",
}

LAYER2_BINARY_COLORS = {
    "TP": "#16a34a",
    "TN": "#86efac",
    "FP": "#dc2626",
    "FN": "#f97316",
}

LAYER2_GRADED_COLORS = {
    "match": "#16a34a",
    "mismatch": "#dc2626",
}

LAYER1_OUTLIER_OUTCOMES = frozenset({"withheld_applicable", "spurious_applicable"})
LAYER2_ERROR_OUTCOMES = frozenset({"FP", "FN", "mismatch"})
