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
