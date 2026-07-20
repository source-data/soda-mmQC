# Append-only log

Add new entries at the **bottom** of this file. Each entry is one `##` heading so tools can list recent activity, for example:

```bash
grep '^## \[' thinking/log.md | tail -5
```

**Prefixes:** `ingest` (material pulled in or summarized), `query` (notable Q&A or decision), `lint` (structure/conventions/wiki hygiene).

---

## [2026-04-30] ingest | Added `index.md` catalog, `log.md`, and README sections for special files and Markdown-vs-HTML guidance.

## [2026-04-30] lint | Removed HTML/CSS wiki mirrors (`index.html`, `evaluation-json-vs-open-source.html`, `wiki.css`); wiki is Markdown-only; README and `index.md` updated.

## [2026-04-30] ingest | Added `evaluation-hierarchical-scoring.md` (roll-up vs drill-down, list-of-X, Hungarian keys, `field_scores` naming); updated `index.md`.

## [2026-04-30] ingest | Added `evaluation-leaf-primitives.md` (schema-driven leaf comparison: strings, numbers, booleans, null roadmap); catalog + README links.

## [2026-04-30] ingest | Expanded `evaluation-leaf-primitives.md` (continuous vs threshold discretization; TP/FP/FN at leaves); linked from `evaluation-hierarchical-scoring.md`.

## [2026-04-30] ingest | Expanded `evaluation-hierarchical-scoring.md`: match/mismatch roll-up, TP/FP/FN for objects, list-of-X, list-of-objects (dual metrics), toy examples, accuracy note.

## [2026-04-30] lint | Clarified wiki vocabulary: `evaluation-leaf-primitives.md` (`score` + conceptual `match` vs parent TP/FP/FN); aligned `evaluation-hierarchical-scoring.md` roll-up / parent metrics wording.

## [2026-04-30] ingest | `evaluation-hierarchical-scoring.md`: toy example Gold `["a","b","c"]` vs Pred `["a","b","x"]` (same-length substitution → one FP slot + one FN slot; `num_mismatches` note).

## [2026-04-30] lint | `evaluation-hierarchical-scoring.md`: fixed object toy (`label` wrong → FP not FN); table for FP vs FN (missing key, `null`, extra key).

## [2026-04-30] ingest | `evaluation-hierarchical-scoring.md`: object FP/FN table is conceptual (JSON Schema for `null`/`""`/missing); removed code-centric slot-counter wording; softened list intro.

## [2026-04-30] ingest | `evaluation-hierarchical-scoring.md`: normative JSON-aligned policy — `""` → FP, `null`/missing key → FN for string-only slots; arguments; list toy de-implementationized. `evaluation-leaf-primitives.md`: null/empty cross-ref.

## [2026-04-30] ingest | `evaluation-hierarchical-scoring.md`: list-of-X when X is object or nested array — element TP/FP/FN vs inner subtree metrics; toys; deduped list-of-objects section.

## [2026-05-21] lint | Harmonized wiki terms: **slot** / **property** / **element** / **value**; removed **edge**; vocabulary section in `evaluation-hierarchical-scoring.md`; aligned `evaluation-leaf-primitives.md`.

## [2026-05-21] ingest | `evaluation-hierarchical-scoring.md`: two-layer reporting — **applicability** (correct_NA, spurious/withheld applicable) vs **answer** (applicable-only; binary yes/no TP/TN; multiclass match/mismatch); scale-bar example; free-text FP/FN scoped separately.

## [2026-05-21] ingest | `evaluation-hierarchical-scoring.md`: **Schema contract** — `enum` alone does not define N/A or polarity; `na_values` / `x-naValues`; string vs boolean; repo examples (`""` vs `not needed`).

## [2026-05-21] ingest | Normative **eval-manifest.json** sidecar (no `x-*` on OpenAI `schema.json`); path-keyed metric profiles; example for micrograph-scale-bar.

## [2026-05-21] ingest | `evaluation-hierarchical-scoring.md`: **Bubbling** — three aggregates (score, structural slots, manifest layers); `whole_object_match` vs applicability/answer F1.

## [2026-05-21] lint | Applicability: separate **`τ_applic`** / `applicability_score`; do not AND spurious_applicable into structural `whole_object_match` (equivalent to `τ_applic=1` only on layer 1).

## [2026-05-21] design | Target **`LeafComparisonResult`** vs **`ComparisonResult`** (wiki only; code reverted — thinking phase).

## [2026-06-08] ingest | `evaluation-toy-examples.md`: shared toy schema/manifest; worked examples for list-of-primitives, object slots, list-of-objects, nested `item.meta`.

## [2026-06-08] ingest | **Flat leaf scoring:** new `evaluation-scoring.md`; rewrote `evaluation-toy-examples.md` (leaf paths only, layer 1/2); `evaluation-hierarchical-scoring.md` marked archive in catalog.

## [2026-06-08] lint | `evaluation-scoring.md`: required `by_property` summaries (mean_score, layer1/2 counts per leaf property); no cross-property aggregation; clarified multi-instance comparisons.

## [2026-06-08] lint | `evaluation-scoring.md` rewritten self-contained: no hierarchical compare/drop table; manifest and scoring rules inlined.

## [2026-06-08] design | List-of-objects alignment: row similarity = mean primitive scores (gold-applicable fields default; fallback all row leaves); alignment-only, not in `by_property`.

## [2026-06-08] design | List-of-objects alignment simplified: manifest `list_alignment` names row field(s) only (e.g. `panels: ["label"]`); removed mean-over-all/applicable heuristics.

## [2026-06-08] lint | Renamed `evaluation-hierarchical-scoring.md` → `evaluation-hierarchical-scoring-DEPRECATED.md`; updated catalog links.

## [2026-06-08] ingest | `evaluation-scoring.md`: manifest `string_compare` (`exact` / `fuzzy` / `semantic`) per field for `graded_string` score vs `match_threshold` for layer 2.

## [2026-06-08] lint | `evaluation-scoring.md`: illustrative manifest — omit `string_compare` on enum `binary_polarity` fields; not in `defaults`; prose on three field classes.

## [2026-06-08] lint | `evaluation-leaf-primitives.md`: aligned with flat `evaluation-scoring.md` — schema vs manifest, `string_compare` scope, no hierarchical roll-up/parent language.

## [2026-06-08] lint | `evaluation-toy-examples.md`: aligned with `evaluation-scoring.md` — assumptions, manifest notes, `by_property` `{}`, D.2 reordered panels, fixed anchor.

## [2026-06-08] ingest | Phase 1 eval: `evaluation-implementation-plan.md`; `soda_mmqc/core/leaves.py` primitive comparators; `tests/test_leaves.py`.

## [2026-06-08] ingest | Fuzzy leaf compare: `rapidfuzz` in `pyproject.toml`; `fuzzy_ratio` replaces hand-rolled LCS in `leaves.py`.

## [2026-06-08] ingest | Phase 2 eval: `eval_manifest.py` (load, profile lookup, list_alignment, path patterns); `tests/test_eval_manifest.py`.

## [2026-06-08] lint | `config.py`: removed legacy `STRING_METRICS` / `DEFAULT_MATCH_THRESHOLD`; `STRING_COMPARE_MODES` only.

## [2026-06-08] lint | Phase 3 module named `applicability_and_answer.py` (was `layers.py`).

## [2026-06-08] lint | Terminology: `answer_metric` → `matching_metric`; module `applicability_and_matching.py`.

## [2026-06-08] ingest | Phase 3 eval: `applicability_and_matching.py`; `tests/test_applicability_and_matching.py` (toy examples A/C/D).

## [2026-06-08] ingest | Phase 4a: `alignment.py` primitive list Hungarian alignment; `tests/test_alignment_primitive_lists.py` (toy B).

## [2026-06-08] design | Layer S structural row alignment (`correct_row`, `missing_row`, `spurious_row`) in `by_list` for list-of-objects only; extended primitives (`tags`) stay aggregate score without layer S; TP/FP/FN reserved for layer 2.

## [2026-06-08] ingest | Wiki: `evaluation-scoring.md`, `evaluation-toy-examples.md`, `evaluation-leaf-primitives.md`, `evaluation-implementation-plan.md` — layer S, `by_list`, extended primitive terminology.

## [2026-06-08] design | Module split: `matching.py`, `leaves.compare_primitive_list`, `object_list_pairing.py`, `structural_reporting.py`; retire `alignment.py`. Phase 4 refresh (4.1–4.4) before phase 5.

## [2026-06-08] ingest | Phase 4.1: `matching.py` (Hungarian, threshold gating); `alignment.py` delegates; `tests/test_matching.py`.

## [2026-06-08] ingest | Phase 4.2: `leaves.compare_primitive_list`, `exact_primitive_similarity`; removed primitive alignment from `alignment.py`; toy B tests in `test_leaves.py`.

## [2026-06-08] ingest | Phase 4.3: `object_list_pairing.py` (`align_object_rows`, pairing-only result); deleted `alignment.py`; `tests/test_object_list_pairing.py`.

## [2026-06-08] ingest | Phase 4.4: `structural_reporting.py` (layer S, `build_by_list`, recall/precision); `tests/test_structural_reporting.py` (toy D.2, D.4, D.5).

## [2026-06-12] design | Structural vs predictive object lists: `list_alignment` marks predictive lists (Hungarian + layer S); structural lists (figures, papers) use positional join, no `by_list`. Wiki updated in `evaluation-scoring.md`.

## [2026-06-12] design | Consolidated list roles: schema is authoritative for predictive (model-produced) vs structural (collection envelope); manifest `list_alignment` configures Hungarian keys and is validated against schema predictive set.

## [2026-06-12] design | Strict schema scope: `schema.json` = model structured-output contract only (no collation wrappers); structural lists exist only in evaluation gold/pred documents.

## [2026-06-12] ingest | Phase 5.1: `collation.py` (eval/schema shape comparison, structural positional join); collated eval tests; demo manifest `figures.panels`.

## [2026-06-12] design | Visualization rewrite plan: flat `analysis.json`, per-property Layer S/1/2 charts, prompt vs model contrast; `thinking/visualization-plan.md`.

## [2026-06-12] design | Visualization plan revised: notebook-first API, drill-down tables first-class (itables), CLI moved to Phase 5.

## [2026-06-12] design | `load_flat_runs`: multi-model + multi-prompt loading; `RunSummaries` keyed by `(model, prompt)`.

## [2026-06-18] impl | Phase 5.2 + 3.1: `property_rollup.py` (mean_score over `correct_applicable`); `plot_mean_score_with_instances` (bar + jittered scatter); tests + `benchmarking.md`.

## [2026-06-16] design | **Layer 2 mean score plot:** separate bar (`mean_score`) + jittered scatter (applicable instance scores); numeric x + `go.Bar`/`go.Scatter`; Phase 3.1 in `visualization-plan.md`.

## [2026-06-16] design | **`mean_score` as Layer 2 rollup:** average instance `score`s over `correct_applicable` only (exclude N/A); normative updates in `evaluation-scoring.md`, `evaluation-leaf-primitives.md`, `visualization-plan.md`, toy examples; implementation **Phase 5.2** in `evaluation-implementation-plan.md`.

## [2026-06-12] design | Run-level `mean_score`: pool all leaf instances per property (no per-figure hierarchical average). *(Superseded for profiled paths by 2026-06-16: pool only `correct_applicable` instances.)*

## [2026-06-12] design | Viz plan: clarify Layer S `by_list` keys (row arrays) vs leaf properties; open decision #2 resolved.

## [2026-06-12] design | Viz v1 scope: single-check reporting only; no checklist-level overview.

## [2026-06-12] design | Viz package name resolved: `soda_mmqc/reporting/`.

## [2026-06-12] design | Interactive drill-down tables: `itables` (resolved).

## [2026-06-12] design | Viz first validation: micrograph-scale-bar with `gpt-5` + `gpt-5-mini`; prompt and model contrast.

## [2026-06-12] ingest | Phase 1 visualization: `soda_mmqc/reporting/` (load, aggregate, tables, styles); `tests/test_reporting_phase1.py`.

## [2026-07-13] impl | Phase 5.3 extended-primitive compare modes: `primitive_list_compare` (`align` / `positional` / `join_string`) in manifest; `compare_primitive_list_positional`, `compare_primitive_list_join` in `leaves.py`; orchestrator branch in `evaluation.py`; tests + `benchmarking.md`.
