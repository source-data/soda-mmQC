"""Phase 4 tests for comparison reporting workflow."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from soda_mmqc.reporting import (
    build_comparison_report,
    load_flat_runs,
    plot_comparison_layer_s,
    show_comparison_report,
    summarize_runs,
)

MODEL_MINI = "gpt-5-mini-2025-08-07"


@pytest.fixture
def summaries_prompt():
    runs = load_flat_runs(
        "fig-checklist",
        "micrograph-scale-bar",
        models=MODEL_MINI,
        prompts=["prompt.1", "prompt.2", "prompt.3"],
    )
    return summarize_runs(runs)


@pytest.fixture
def summaries_model():
    runs = load_flat_runs(
        "fig-checklist",
        "micrograph-scale-bar",
        models=[MODEL_MINI, "gpt-5"],
        prompts="prompt.1",
    )
    return summarize_runs(runs)


class TestComparisonReport:
    def test_build_prompt_contrast_report(self, summaries_prompt):
        report = build_comparison_report(
            summaries_prompt,
            compare="prompt",
            model=MODEL_MINI,
        )
        assert report.compare == "prompt"
        assert report.anchor == MODEL_MINI
        assert report.series_labels == ("prompt.1", "prompt.2", "prompt.3")
        assert report.layer_s_figure is not None
        assert len(report.layer1_figure.data) > 0
        assert len(report.layer2_binary_figure.data) > 0
        assert len(report.layer2_graded_figure.data) > 0
        assert "prompt" in report.errors_table.columns

    def test_build_model_contrast_report(self, summaries_model):
        report = build_comparison_report(
            summaries_model,
            compare="model",
            prompt="prompt.1",
        )
        assert report.compare == "model"
        assert report.anchor == "prompt.1"
        assert report.series_labels == (MODEL_MINI, "gpt-5")
        assert "model" in report.errors_table.columns

    def test_prompt_contrast_scale_bar_signal(self, summaries_prompt):
        report = build_comparison_report(
            summaries_prompt,
            compare="prompt",
            model=MODEL_MINI,
        )
        prompt2_errors = report.errors_table.loc[
            report.errors_table["prompt"] == "prompt.2"
        ]
        assert not prompt2_errors.empty
        assert prompt2_errors["leaf_property"].str.contains("scale_bar").any()

    def test_plot_comparison_layer_s_series(self, summaries_prompt):
        fig = plot_comparison_layer_s(
            summaries_prompt,
            compare="prompt",
            model=MODEL_MINI,
        )
        assert fig is not None
        assert len(fig.data) == 3
        opacities = [trace.marker.opacity for trace in fig.data]
        assert len(set(opacities)) == 3

    def test_show_comparison_report_returns_report(self, summaries_model):
        with patch("plotly.graph_objects.Figure.show"):
            report = show_comparison_report(
                summaries_model,
                compare="model",
                prompt="prompt.1",
                show_errors_table=False,
            )
        assert report.compare == "model"
        assert len(report.errors_table.columns) > 0

    def test_requires_selector(self, summaries_prompt):
        with pytest.raises(ValueError, match="model is required"):
            build_comparison_report(summaries_prompt, compare="prompt")
