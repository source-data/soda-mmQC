"""Evaluate quote alignments in mmQC JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Any

from mmqc_utils import AlignmentGap, CharInterval, QuoteAlignment, align_quote, compute_plain_text

STATUS_SORT_ORDER = {
    "none": 0,
    "match_fuzzy": 1,
    "match_lesser": 2,
    "match_greater": 3,
    "match_exact": 4,
}

EXTRACTION_TYPE_ORDER = {
    "caption_section": 0,
    "data_availability": 1,
    "figure_caption": 2,
    "panel_caption": 3,
}


@dataclass(frozen=True)
class AlignmentCase:
    case_id: str
    source_file: str
    manuscript_id: str
    extraction_type: str
    quote_label: str
    quote: str
    text: str


@dataclass(frozen=True)
class AlignmentResult:
    case: AlignmentCase
    alignment: QuoteAlignment
    plain_quote: str
    plain_text: str

    @property
    def status_value(self) -> str:
        if self.alignment.alignment_status is None:
            return "none"
        return self.alignment.alignment_status.value


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if not any(key in data for key in ("manuscript_text", "ai_response_locate_captions", "figures")):
        return None
    return data


def _str_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def collect_cases(input_dir: Path) -> list[AlignmentCase]:
    cases: list[AlignmentCase] = []
    for path in sorted(input_dir.glob("*.json")):
        data = _read_json(path)
        if data is None:
            continue

        manuscript_id = _str_value(data.get("manuscript_id")) or path.name
        manuscript_text = _str_value(data.get("manuscript_text"))
        captions_text = _str_value(data.get("ai_response_locate_captions"))

        cases.append(
            AlignmentCase(
                case_id=f"{manuscript_id}::caption_section",
                source_file=path.name,
                manuscript_id=manuscript_id,
                extraction_type="caption_section",
                quote_label="ai_response_locate_captions",
                quote=captions_text,
                text=manuscript_text,
            )
        )

        data_availability_raw = data.get("data_availability")
        data_availability = data_availability_raw if isinstance(data_availability_raw, dict) else {}
        cases.append(
            AlignmentCase(
                case_id=f"{manuscript_id}::data_availability",
                source_file=path.name,
                manuscript_id=manuscript_id,
                extraction_type="data_availability",
                quote_label="data_availability.section_text",
                quote=_str_value(data_availability.get("section_text")),
                text=manuscript_text,
            )
        )

        for figure_index, figure in enumerate(data.get("figures") or [], start=1):
            if not isinstance(figure, dict):
                continue
            figure_label = _str_value(figure.get("figure_label")) or f"Figure {figure_index}"
            figure_caption = _str_value(figure.get("figure_caption"))

            cases.append(
                AlignmentCase(
                    case_id=f"{manuscript_id}::{figure_label}::figure_caption",
                    source_file=path.name,
                    manuscript_id=manuscript_id,
                    extraction_type="figure_caption",
                    quote_label=figure_label,
                    quote=figure_caption,
                    text=captions_text,
                )
            )

            for panel_index, panel in enumerate(figure.get("panels") or [], start=1):
                if not isinstance(panel, dict):
                    continue
                panel_label = _str_value(panel.get("panel_label")) or f"panel-{panel_index}"
                cases.append(
                    AlignmentCase(
                        case_id=f"{manuscript_id}::{figure_label}::{panel_label}::panel_caption",
                        source_file=path.name,
                        manuscript_id=manuscript_id,
                        extraction_type="panel_caption",
                        quote_label=f"{figure_label} {panel_label}",
                        quote=_str_value(panel.get("panel_caption")),
                        text=figure_caption,
                    )
                )

    return cases


def evaluate_cases(cases: list[AlignmentCase]) -> list[AlignmentResult]:
    results: list[AlignmentResult] = []
    for case in cases:
        plain_quote = compute_plain_text(case.quote)
        plain_text = compute_plain_text(case.text)
        alignment = align_quote(plain_quote, plain_text)
        results.append(
            AlignmentResult(
                case=case,
                alignment=alignment,
                plain_quote=plain_quote,
                plain_text=plain_text,
            )
        )
    return results


def _intervals_to_dicts(intervals: list[CharInterval] | None) -> list[dict[str, int | None]]:
    if intervals is None:
        return []
    return [{"start_pos": interval.start_pos, "end_pos": interval.end_pos} for interval in intervals]


def _gaps_to_dicts(gaps: list[AlignmentGap] | None) -> list[dict[str, Any]]:
    if gaps is None:
        return []
    return [
        {
            "text": gap.text,
            "char_interval": {
                "start_pos": gap.char_interval.start_pos,
                "end_pos": gap.char_interval.end_pos,
            },
            "length": len(gap.text),
        }
        for gap in gaps
    ]


def _gap_chars(gaps: list[AlignmentGap] | None) -> int:
    return sum(len(gap.text) for gap in gaps or [])


def write_json(results: list[AlignmentResult], output_path: Path) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "case_id": result.case.case_id,
                "source_file": result.case.source_file,
                "manuscript_id": result.case.manuscript_id,
                "extraction_type": result.case.extraction_type,
                "quote_label": result.case.quote_label,
                "score": result.alignment.score,
                "alignment_status": result.status_value,
                "char_intervals": _intervals_to_dicts(result.alignment.char_intervals),
                "source_gaps": _gaps_to_dicts(result.alignment.source_gaps),
                "quote_gaps": _gaps_to_dicts(result.alignment.quote_gaps),
                "plain_quote_length": len(result.plain_quote),
                "plain_text_length": len(result.plain_text),
                "matched_text": result.alignment.matched_text,
            }
        )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(results: list[AlignmentResult], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "source_file",
                "manuscript_id",
                "extraction_type",
                "quote_label",
                "score",
                "alignment_status",
                "interval_count",
                "source_gap_count",
                "source_gap_chars",
                "quote_gap_count",
                "quote_gap_chars",
                "plain_quote_length",
                "plain_text_length",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case.case_id,
                    "source_file": result.case.source_file,
                    "manuscript_id": result.case.manuscript_id,
                    "extraction_type": result.case.extraction_type,
                    "quote_label": result.case.quote_label,
                    "score": f"{result.alignment.score:.6f}",
                    "alignment_status": result.status_value,
                    "interval_count": len(result.alignment.char_intervals or []),
                    "source_gap_count": len(result.alignment.source_gaps or []),
                    "source_gap_chars": _gap_chars(result.alignment.source_gaps),
                    "quote_gap_count": len(result.alignment.quote_gaps or []),
                    "quote_gap_chars": _gap_chars(result.alignment.quote_gaps),
                    "plain_quote_length": len(result.plain_quote),
                    "plain_text_length": len(result.plain_text),
                }
            )


def _scores(results: list[AlignmentResult]) -> list[float]:
    return [result.alignment.score for result in results]


def _score_summary(results: list[AlignmentResult]) -> dict[str, float | int]:
    scores = _scores(results)
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(scores),
        "mean": mean(scores),
        "median": median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def _group_by(results: list[AlignmentResult], key: str) -> dict[str, list[AlignmentResult]]:
    grouped: dict[str, list[AlignmentResult]] = defaultdict(list)
    for result in results:
        if key == "type":
            value = result.case.extraction_type
        elif key == "manuscript":
            value = result.case.manuscript_id
        elif key == "status":
            value = result.status_value
        else:
            raise ValueError(f"Unknown grouping key: {key}")
        grouped[value].append(result)
    return dict(sorted(grouped.items()))


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{100 * numerator / denominator:.1f}%"


def _format_float(value: float | int) -> str:
    return f"{float(value):.3f}"


def _format_score(value: float | int) -> str:
    return f"{float(value):.6f}"


def _case_sort_key(result: AlignmentResult) -> tuple[float, int, str, str, str]:
    return (
        result.alignment.score,
        STATUS_SORT_ORDER.get(result.status_value, 99),
        result.case.manuscript_id,
        result.case.quote_label,
        result.case.case_id,
    )


def _is_perfect_html_match(result: AlignmentResult) -> bool:
    if result.status_value == "match_exact":
        return True
    return result.status_value == "match_greater" and result.alignment.score == 1.0


def _snippet(text: str, interval: CharInterval, context: int = 240) -> str:
    if interval.start_pos is None or interval.end_pos is None:
        return ""
    start = max(0, interval.start_pos - context)
    end = min(len(text), interval.end_pos + context)
    before = escape(text[start : interval.start_pos])
    match = escape(text[interval.start_pos : interval.end_pos])
    after = escape(text[interval.end_pos : end])
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{before}<mark>{match}</mark>{after}{suffix}"


def _preview(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return escape(text)
    return escape(text[:limit]) + "..."


def _summary_table(title: str, grouped: dict[str, list[AlignmentResult]]) -> str:
    rows = []
    for label, group in grouped.items():
        summary = _score_summary(group)
        status_counts = Counter(result.status_value for result in group)
        intervals = sum(1 for result in group if result.alignment.char_intervals)
        rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{summary['count']}</td>"
            f"<td>{_format_float(summary['mean'])}</td>"
            f"<td>{_format_float(summary['median'])}</td>"
            f"<td>{_format_float(summary['min'])}</td>"
            f"<td>{_format_float(summary['max'])}</td>"
            f"<td>{_pct(intervals, int(summary['count']))}</td>"
            f"<td>{escape(', '.join(f'{k}: {v}' for k, v in sorted(status_counts.items())))}</td>"
            "</tr>"
        )
    return (
        f"<h2>{escape(title)}</h2>"
        "<table><thead><tr><th>Group</th><th>n</th><th>Mean</th><th>Median</th>"
        "<th>Min</th><th>Max</th><th>With intervals</th><th>Status counts</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _case_table(title: str, items: list[tuple[int, AlignmentResult, str]]) -> str:
    rows = []
    for index, result, anchor in items:
        intervals = result.alignment.char_intervals or []
        rows.append(
            "<tr>"
            f"<td><a href='#{anchor}'>{index}</a></td>"
            f"<td>{escape(result.case.manuscript_id)}</td>"
            f"<td>{escape(result.case.quote_label)}</td>"
            f"<td>{_format_score(result.alignment.score)}</td>"
            f"<td>{escape(result.status_value)}</td>"
            f"<td>{len(intervals)}</td>"
            f"<td>{len(result.alignment.source_gaps or [])}</td>"
            f"<td>{len(result.alignment.quote_gaps or [])}</td>"
            f"<td>{len(result.plain_quote)}</td>"
            f"<td>{len(result.plain_text)}</td>"
            "</tr>"
        )
    return (
        f"<h3>{escape(title)}</h3>"
        "<table><thead><tr><th>#</th><th>Manuscript</th><th>Label</th><th>Score</th>"
        "<th>Status</th><th>Intervals</th><th>Source gaps</th><th>Quote gaps</th>"
        "<th>Quote chars</th><th>Source chars</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _gap_details(title: str, gaps: list[AlignmentGap] | None) -> str:
    if not gaps:
        return f"<h4>{escape(title)}</h4><p class='muted'>None.</p>"

    details = []
    for index, gap in enumerate(gaps, start=1):
        interval = gap.char_interval
        details.append(
            "<details>"
            f"<summary>Gap {index}: {interval.start_pos}-{interval.end_pos}, {len(gap.text)} chars</summary>"
            f"<pre>{_preview(gap.text)}</pre>"
            "</details>"
        )
    return f"<h4>{escape(title)}</h4>{''.join(details)}"


def _source_text_details(result: AlignmentResult) -> str:
    intervals = result.alignment.char_intervals or []
    if result.case.extraction_type != "panel_caption" and intervals:
        return ""

    title = "Figure caption source text" if result.case.extraction_type == "panel_caption" else "Source text"
    open_attr = " open" if not intervals else ""
    return (
        f"<details class='source-text'{open_attr}>"
        f"<summary>{escape(title)}: {len(result.plain_text)} chars</summary>"
        f"<pre>{escape(result.plain_text)}</pre>"
        "</details>"
    )


def write_html(results: list[AlignmentResult], output_path: Path, *, exclude_perfect: bool = False) -> None:
    display_results = (
        [result for result in results if not _is_perfect_html_match(result)] if exclude_perfect else results
    )
    hidden_count = len(results) - len(display_results)
    overall = _score_summary(display_results)
    status_counts = Counter(result.status_value for result in display_results)
    with_intervals = sum(1 for result in display_results if result.alignment.char_intervals)
    exact_ones = sum(1 for result in display_results if result.alignment.score == 1.0)
    filter_note = (
        f"<p class='muted'>HTML filter active: hidden {hidden_count} exact/perfect-greater cases. "
        "JSON and CSV still contain all evaluated cases.</p>"
        if exclude_perfect
        else ""
    )

    ordered_results = sorted(display_results, key=_case_sort_key)
    ordered_items = [(index, result, f"case-{index}") for index, result in enumerate(ordered_results, start=1)]
    items_by_type: dict[str, list[tuple[int, AlignmentResult, str]]] = defaultdict(list)
    for item in ordered_items:
        items_by_type[item[1].case.extraction_type].append(item)
    sorted_type_items = sorted(
        items_by_type.items(), key=lambda item: (EXTRACTION_TYPE_ORDER.get(item[0], 99), item[0])
    )

    case_tables = [_case_table(extraction_type, type_items) for extraction_type, type_items in sorted_type_items]
    detail_sections = []
    for index, result, anchor in ordered_items:
        intervals = result.alignment.char_intervals or []
        interval_html = ""
        if intervals:
            snippets = []
            for interval_index, interval in enumerate(intervals, start=1):
                snippets.append(
                    f"<h4>Interval {interval_index}: {interval.start_pos}-{interval.end_pos}</h4>"
                    f"<pre class='snippet'>{_snippet(result.plain_text, interval)}</pre>"
                )
            interval_html = "".join(snippets)
        else:
            interval_html = "<p class='muted'>No literal intervals returned. Fuzzy fallback only.</p>"

        detail_sections.append(
            f"<section class='case' id='{anchor}'>"
            f"<h3>{index}. {escape(result.case.manuscript_id)} · {escape(result.case.extraction_type)} · "
            f"{escape(result.case.quote_label)}</h3>"
            "<dl>"
            f"<dt>Source file</dt><dd>{escape(result.case.source_file)}</dd>"
            f"<dt>Score</dt><dd>{_format_score(result.alignment.score)}</dd>"
            f"<dt>Status</dt><dd>{escape(result.status_value)}</dd>"
            f"<dt>Source gaps</dt><dd>{len(result.alignment.source_gaps or [])} "
            f"({_gap_chars(result.alignment.source_gaps)} chars)</dd>"
            f"<dt>Quote gaps</dt><dd>{len(result.alignment.quote_gaps or [])} "
            f"({_gap_chars(result.alignment.quote_gaps)} chars)</dd>"
            f"<dt>Plain quote length</dt><dd>{len(result.plain_quote)}</dd>"
            f"<dt>Plain text length</dt><dd>{len(result.plain_text)}</dd>"
            "</dl>"
            "<h4>Quote</h4>"
            f"<pre>{escape(result.plain_quote)}</pre>"
            f"{_source_text_details(result)}"
            "<h4>Alignment</h4>"
            f"{interval_html}"
            f"{_gap_details('Source gaps', result.alignment.source_gaps)}"
            f"{_gap_details('Quote gaps', result.alignment.quote_gaps)}"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Quote Alignment Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; position: sticky; top: 0; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f8fafc;
      border: 1px solid #d9e2ec;
      padding: 12px;
    }}
    mark {{ background: #ffe066; padding: 1px 2px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .card {{ border: 1px solid #d9e2ec; padding: 12px; background: #f8fafc; }}
    .metric {{ font-size: 24px; font-weight: 700; }}
    .muted {{ color: #627d98; }}
    .case {{ border-top: 2px solid #bcccdc; padding-top: 20px; margin-top: 28px; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; }}
    dt {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Quote Alignment Report</h1>
  <div class="cards">
    <div class="card"><div>Evaluated cases</div><div class="metric">{len(results)}</div></div>
    <div class="card"><div>Displayed cases</div><div class="metric">{overall["count"]}</div></div>
    <div class="card"><div>Hidden cases</div><div class="metric">{hidden_count}</div></div>
    <div class="card"><div>Mean score</div><div class="metric">{_format_float(overall["mean"])}</div></div>
    <div class="card"><div>Median score</div><div class="metric">{_format_float(overall["median"])}</div></div>
    <div class="card">
      <div>With intervals</div>
      <div class="metric">{_pct(with_intervals, int(overall["count"]))}</div>
    </div>
    <div class="card"><div>Score = 1.0</div><div class="metric">{_pct(exact_ones, int(overall["count"]))}</div></div>
  </div>
  {filter_note}
  <p>Status counts: {escape(", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())))}</p>
  {_summary_table("By Extraction Type", _group_by(display_results, "type"))}
  {_summary_table("By Manuscript", _group_by(display_results, "manuscript"))}
  <h2>Cases</h2>
  <p class="muted">Each table is sorted by score ascending, then status.</p>
  {"".join(case_tables)}
  <h2>Drilldown</h2>
  {"".join(detail_sections)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing JSON files to evaluate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quote_alignment_report"),
        help="Output directory for report files. Defaults to ./quote_alignment_report.",
    )
    parser.add_argument("--html-name", default="quote_alignment_report.html")
    parser.add_argument("--json-name", default="quote_alignment_results.json")
    parser.add_argument("--csv-name", default="quote_alignment_results.csv")
    parser.add_argument(
        "--html-exclude-perfect",
        action="store_true",
        help="Exclude match_exact and score=1.0 match_greater cases from the HTML report only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(input_dir)
    results = evaluate_cases(cases)
    write_html(results, output_dir / args.html_name, exclude_perfect=args.html_exclude_perfect)
    write_json(results, output_dir / args.json_name)
    write_csv(results, output_dir / args.csv_name)

    summary = _score_summary(results)
    print(f"Loaded {len(cases)} cases from {input_dir}")
    print(f"Mean score: {_format_float(summary['mean'])}")
    print(f"HTML report: {output_dir / args.html_name}")
    print(f"JSON results: {output_dir / args.json_name}")
    print(f"CSV results: {output_dir / args.csv_name}")


if __name__ == "__main__":
    main()
