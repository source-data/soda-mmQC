"""Tests for evaluation check discovery and Streamlit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soda_mmqc.reporting.display import build_figure_image_plot
from soda_mmqc.reporting.load import discover_evaluation_checks, try_load_run_summaries
from soda_mmqc.reporting.plots import plot_mean_score_with_instances
from soda_mmqc.reporting.streamlit_app import _selected_instance_index


class TestDiscoverEvaluationChecks:
    def test_discovers_micrograph_scale_bar(self):
        refs = discover_evaluation_checks()
        assert any(
            ref.checklist == "fig-checklist" and ref.check == "micrograph-scale-bar"
            for ref in refs
        )

    def test_empty_when_no_evaluation_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            "soda_mmqc.reporting.load.EVALUATION_DIR",
            tmp_path / "missing",
        )
        assert discover_evaluation_checks() == ()

    def test_finds_check_with_analysis_json(self, tmp_path: Path, monkeypatch):
        eval_root = tmp_path / "evaluation"
        analysis_path = (
            eval_root / "fig-checklist" / "demo-check" / "gpt-5" / "analysis.json"
        )
        analysis_path.parent.mkdir(parents=True)
        analysis_path.write_text(json.dumps({"prompt.1": {"flat": []}}), encoding="utf-8")
        monkeypatch.setattr("soda_mmqc.reporting.load.EVALUATION_DIR", eval_root)
        refs = discover_evaluation_checks()
        assert len(refs) == 1
        assert refs[0].checklist == "fig-checklist"
        assert refs[0].check == "demo-check"


class TestMeanScorePlotSelection:
    def test_return_instances_includes_customdata(self, prompt1_summary):
        fig, inst = plot_mean_score_with_instances(
            prompt1_summary,
            return_instances=True,
        )
        scatter = next(trace for trace in fig.data if trace.type == "scatter")
        assert len(scatter.customdata) == len(inst)
        assert "source" in inst.columns


class TestFigureImagePlot:
    def test_build_figure_image_plot(self):
        image_path = (
            Path(__file__).resolve().parents[1]
            / "soda_mmqc/data/examples/10.1038_s44318-026-00715-1/content/1"
            / "content/1.png"
        )
        if not image_path.is_file():
            pytest.skip("figure fixture image missing")
        fig = build_figure_image_plot(image_path, height=200)
        assert fig is not None
        assert fig.layout.dragmode == "zoom"


class TestTryLoadRunSummaries:
    def test_missing_manifest_returns_message(self, tmp_path: Path, monkeypatch):
        eval_root = tmp_path / "evaluation"
        check_eval = eval_root / "fig-checklist" / "demo-check"
        (check_eval / "gpt-5").mkdir(parents=True)
        (check_eval / "gpt-5" / "analysis.json").write_text(
            json.dumps({"prompt.1": {"flat": []}}),
            encoding="utf-8",
        )
        checklist_root = tmp_path / "checklist" / "fig-checklist" / "demo-check"
        checklist_root.mkdir(parents=True)

        monkeypatch.setattr("soda_mmqc.reporting.load.EVALUATION_DIR", eval_root)
        monkeypatch.setattr("soda_mmqc.reporting.load.CHECKLIST_DIR", tmp_path / "checklist")

        summaries, error = try_load_run_summaries("fig-checklist", "demo-check")
        assert summaries is None
        assert error is not None
        assert "eval-manifest.json" in error

    def test_loads_when_manifest_present(self, tmp_path: Path, monkeypatch):
        eval_root = tmp_path / "evaluation"
        check_eval = eval_root / "fig-checklist" / "demo-check"
        (check_eval / "gpt-5").mkdir(parents=True)
        (check_eval / "gpt-5" / "analysis.json").write_text(
            json.dumps({"prompt.1": {"flat": []}}),
            encoding="utf-8",
        )
        checklist_root = tmp_path / "checklist" / "fig-checklist" / "demo-check"
        checklist_root.mkdir(parents=True)
        (checklist_root / "eval-manifest.json").write_text(
            json.dumps(
                {
                    "checklist": "fig-checklist",
                    "check": "demo-check",
                    "defaults": {},
                    "fields": {},
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr("soda_mmqc.reporting.load.EVALUATION_DIR", eval_root)
        monkeypatch.setattr("soda_mmqc.reporting.load.CHECKLIST_DIR", tmp_path / "checklist")

        summaries, error = try_load_run_summaries("fig-checklist", "demo-check")
        assert error is None
        assert summaries is not None
        assert len(summaries) == 1

    def test_loads_image_annotation_defined(self):
        summaries, error = try_load_run_summaries(
            "fig-checklist",
            "image-annotation-defined",
        )
        assert error is None, error
        assert summaries is not None
        assert len(summaries) > 0
        assert "gpt-5-mini-2025-08-07" in summaries.models


class TestStreamlitSelectionParsing:
    def test_selected_instance_index_from_customdata(self):
        class Selection:
            points = [{"customdata": [3]}]

        class Event:
            selection = Selection()

        assert _selected_instance_index(Event().selection) == 3

    def test_selected_instance_index_empty(self):
        assert _selected_instance_index(None) is None
        assert _selected_instance_index(type("S", (), {"points": []})()) is None


@pytest.fixture
def prompt1_summary():
    from soda_mmqc.reporting import aggregate_run, load_flat_runs

    runs = load_flat_runs(
        "fig-checklist",
        "micrograph-scale-bar",
        models="gpt-5-mini-2025-08-07",
        prompts="prompt.1",
    )
    return aggregate_run(runs[0])
