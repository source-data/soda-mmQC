# analysis_json_to_html

Helper to convert `analysis.json` evaluation outputs into human-readable `analysis.html` pages.

This improved renderer groups findings per manuscript (document) and presents:
- A concise overall verdict (Reproducible / Partially reproducible / Not reproducible)
- Per-panel reasonings and discrepancies (short human-readable text)
- References to the original `analysis.json` location and any referenced tables (no full JSON is embedded)

Usage

Run from the repository root. Examples:

```bash
python soda_mmqc/scripts/analysis_json_to_html.py --analysis-path "soda_mmqc/data/evaluation/fig-checklist/panel-data-replication-validation/gpt-5-mini-2025-08-07/analysis.json"

python soda_mmqc/scripts/analysis_json_to_html.py --evaluation panel-data-replication-validation

python soda_mmqc/scripts/analysis_json_to_html.py --evaluation panel-data-replication-validation --group fig-checklist
```

Output

For each `analysis.json` processed the script writes an `analysis.html` file next to it. The HTML contains per-manuscript sections with verdicts and the reasonings for each panel; the full JSON/prompt locations are shown as paths rather than embedded content.
