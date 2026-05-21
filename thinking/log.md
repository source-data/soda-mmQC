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
