"""Phase 2 tests for soda_mmqc.reporting.display."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from soda_mmqc.reporting import (
    comparison_errors_table,
    load_flat_runs,
    show_comparison_errors,
    show_layer1_errors,
    show_layer2_errors,
    show_table,
    sort_frame,
    summarize_runs,
)
from soda_mmqc.reporting.display import (
    _column_search_cols,
    _itables_available,
    sort_frame as sort_frame_fn,
)


class TestSortFrame:
    def test_sorts_by_default_columns(self):
        frame = pd.DataFrame(
            {"doc_id": ["b", "a"], "field": ["y", "x"], "path": ["p2", "p1"]}
        )
        sorted_frame = sort_frame(frame, ("doc_id", "field"))
        assert sorted_frame["doc_id"].tolist() == ["a", "b"]


class TestComparisonErrorsTable:
    @pytest.fixture
    def summaries(self):
        runs = load_flat_runs(
            "fig-checklist",
            "micrograph-scale-bar",
            models="gpt-5-mini-2025-08-07",
            prompts=["prompt.1", "prompt.2"],
        )
        return summarize_runs(runs)

    def test_prompt_contrast_includes_prompt_column(self, summaries):
        frame = comparison_errors_table(
            summaries,
            compare="prompt",
            model="gpt-5-mini-2025-08-07",
        )
        if frame.empty:
            pytest.skip("no layer-2 errors in fixture run")
        assert "prompt" in frame.columns
        assert set(frame["prompt"].unique()).issubset({"prompt.1", "prompt.2"})

    def test_prompt2_scale_bar_culprits(self, summaries):
        frame = comparison_errors_table(
            summaries,
            compare="prompt",
            model="gpt-5-mini-2025-08-07",
        )
        prompt2 = frame.loc[frame["prompt"] == "prompt.2"]
        assert not prompt2.empty
        assert prompt2["leaf_property"].str.contains("scale_bar").any()

    def test_model_contrast_requires_prompt(self, summaries):
        with pytest.raises(ValueError, match="prompt is required"):
            comparison_errors_table(summaries, compare="model")


class TestColumnSearchCols:
    def test_builds_search_cols_aligned_to_frame(self):
        frame = pd.DataFrame({"layer1": ["a"], "field": ["x"]})
        assert _column_search_cols(frame, {"layer1": "spurious_applicable"}) == [
            {"search": "spurious_applicable"},
            None,
        ]


class TestShowTable:
    def test_show_table_uses_itables_when_available(self):
        frame = pd.DataFrame({"a": [1]})
        if not _itables_available():
            pytest.skip("itables not installed")

        with patch("itables.show", return_value="widget") as mock_show:
            result = show_table(frame, caption="test", default_sort=("a",))
        mock_show.assert_called_once()
        assert result == "widget"

    def test_show_table_passes_column_search(self):
        frame = pd.DataFrame({"layer1": ["spurious_applicable"], "field": ["x"]})
        if not _itables_available():
            pytest.skip("itables not installed")

        with patch("itables.show", return_value="widget") as mock_show:
            show_table(frame, column_search={"layer1": "spurious_applicable"})
        assert mock_show.call_args.kwargs["searchCols"] == [
            {"search": "spurious_applicable"},
            None,
        ]

    def test_show_table_fallback_without_itables(self):
        frame = pd.DataFrame({"doc_id": ["x"], "field": ["micrograph"]})
        with patch("soda_mmqc.reporting.display._itables_available", return_value=False):
            with patch("IPython.display.display") as mock_display:
                result = show_table(frame)
        assert isinstance(result, pd.DataFrame)
        assert mock_display.call_count >= 1

    def test_show_layer1_errors_presets_layer1_column_search(self):
        runs = load_flat_runs(
            "fig-checklist",
            "micrograph-scale-bar",
            models="gpt-5",
            prompts="prompt.2",
        )
        summaries = summarize_runs(runs)
        summary = summaries["gpt-5", "prompt.2"]

        with patch("soda_mmqc.reporting.display.show_table") as mock_show:
            show_layer1_errors(summary, layer1="spurious_applicable")
        assert mock_show.call_args.kwargs["column_search"] == {
            "layer1": "spurious_applicable",
        }
        passed_frame = mock_show.call_args[0][0]
        assert "layer1" in passed_frame.columns

    def test_show_layer2_errors_filters_field(self):
        runs = load_flat_runs(
            "fig-checklist",
            "micrograph-scale-bar",
            models="gpt-5-mini-2025-08-07",
            prompts="prompt.2",
        )
        summaries = summarize_runs(runs)
        summary = summaries["gpt-5-mini-2025-08-07", "prompt.2"]

        with patch("soda_mmqc.reporting.display.show_table") as mock_show:
            show_layer2_errors(summary, field="scale_bar_on_image")
        passed_frame = mock_show.call_args[0][0]
        assert not passed_frame.empty
        assert (passed_frame["leaf_property"] == "scale_bar_on_image").all()

    def test_show_comparison_errors_delegates(self):
        runs = load_flat_runs(
            "fig-checklist",
            "micrograph-scale-bar",
            models="gpt-5-mini-2025-08-07",
            prompts=["prompt.1", "prompt.2"],
        )
        summaries = summarize_runs(runs)
        with patch("soda_mmqc.reporting.display.show_table") as mock_show:
            show_comparison_errors(
                summaries,
                compare="prompt",
                model="gpt-5-mini-2025-08-07",
            )
        mock_show.assert_called_once()
        passed_frame = mock_show.call_args[0][0]
        assert "prompt" in passed_frame.columns
