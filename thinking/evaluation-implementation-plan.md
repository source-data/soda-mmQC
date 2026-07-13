# Evaluation implementation plan (progressive)

Normative contract: [evaluation-scoring.md](evaluation-scoring.md). Worked examples: [evaluation-toy-examples.md](evaluation-toy-examples.md).

Build the flat leaf comparator in small steps. Each step lands with **unit tests** before the next layer depends on it.

**Module layout** (target after phase 4 refresh):

| Module | Role |
|--------|------|
| [`leaves.py`](../soda_mmqc/core/leaves.py) | Pairwise primitive compare; public **`compare_primitive_list`** → `LeafComparisonResult` |
| [`matching.py`](../soda_mmqc/core/matching.py) | Shared bipartite matching (similarity matrix, Hungarian, threshold gating) — no domain semantics |
| [`object_list_pairing.py`](../soda_mmqc/core/object_list_pairing.py) | Row pairing for list-of-objects (`list_alignment` keys); feeds layer S |
| [`structural_reporting.py`](../soda_mmqc/core/structural_reporting.py) | Layer S → `by_list` (`correct_row`, `missing_row`, `spurious_row`) |
| [`eval_manifest.py`](../soda_mmqc/core/eval_manifest.py) | Manifest load and field profiles |
| [`applicability_and_matching.py`](../soda_mmqc/core/applicability_and_matching.py) | Layers 1 and 2 reporting on leaf instances |
| [`schema_discovery.py`](../soda_mmqc/core/schema_discovery.py) | Model schema walk → leaf properties + predictive list names |
| [`collation.py`](../soda_mmqc/core/collation.py) | Embed schema in eval JSON; structural vs predictive paths; manifest validation |
| [`evaluation.py`](../soda_mmqc/core/evaluation.py) | Orchestrator |

`alignment.py` has been **removed** (split into `matching.py`, `leaves.compare_primitive_list`, and `object_list_pairing.py`).

**Dependency direction:** `matching` ← `leaves`, `object_list_pairing`; `structural_reporting` ← `object_list_pairing`; `evaluation` ← all of the above. `leaves` does not import `object_list_pairing`.

---

## Phase 1 — Scalar primitive leaf comparison ✅ *complete*

**Module:** `soda_mmqc/core/leaves.py`

Pure functions: two scalar values (+ minimal schema hints) → `LeafComparisonResult` with **`score ∈ [0, 1]`** only. No manifest layers, no list matching, no `match_threshold` (that belongs in the outer comparator).

| Function | Responsibility |
|----------|----------------|
| `fuzzy_ratio` (rapidfuzz) | Fuzzy string helper |
| `compare_exact_strings` | Character-level equality |
| `compare_fuzzy_strings` | rapidfuzz `fuzz.ratio` (normalized) |
| `compare_semantic_strings` | Embedding cosine similarity (lazy model; injectable for tests) |
| `compare_enum_string` | Pred ∈ `enum`; exact literal vs gold |
| `compare_boolean` | Exact equality |
| `compare_number` | Exact equality (`int` / `float`) |
| `compare_strings` | Dispatch on `StringCompareMode` (`exact` / `fuzzy` / `semantic`) |

**Tests:** `tests/test_leaves.py` — one test class (or module section) per function.

**Out of scope for phase 1:** extended primitives (`compare_primitive_list`), orchestrator output.

---

## Phase 2 — Manifest model ✅ *complete*

**Module:** `soda_mmqc/core/eval_manifest.py`

- `load_eval_manifest` / `parse_eval_manifest`
- `FieldProfile`, `EvalManifest`, `MatchingMetric`
- `profile_for(leaf_property)` with defaults inheritance (key-aware merge)
- `list_alignment` top-level map + `alignment_keys_for`
- Path helpers: `instance_to_leaf_property`, `path_matches_pattern`
- Validation: no `string_compare` / `match_threshold` in defaults; graded_string and binary_polarity rules

**Tests:** `tests/test_eval_manifest.py` + `tests/fixtures/eval_manifest_toy.json`

---

## Phase 3 — Applicability and matching reporting ✅ *complete*

**Module:** `soda_mmqc/core/applicability_and_matching.py`

Given `LeafComparisonResult` + field profile + `(exp, pred)` values:

- **Layer 1 (applicability) reporting:** `correct_NA`, `spurious_applicable`, `withheld_applicable`, `correct_applicable`
- **Layer 2 (matching) reporting:** `TP`/`FP`/`FN`/`TN`, `match`/`mismatch` per `matching_metric` (value-level only)

**Tests:** `tests/test_applicability_and_matching.py` — mirror [evaluation-toy-examples.md](evaluation-toy-examples.md) sections C (status/label) and graded_string threshold cases.

Phases 1–3 require **no code changes** for the layer S / module-split design.

---

## Phase 4 refresh — matching split, extended primitives, object pairing, layer S

Replaces the earlier monolithic `alignment.py` phases 4a/4b. Implement in order.

### Phase 4.1 — Shared matching (`matching.py`) ✅ *complete*

**Module:** `soda_mmqc/core/matching.py`

- `build_similarity_matrix`, `hungarian_assignment`, `hungarian_match_pairs`
- `pairs_at_threshold`, `mean_gated_similarity`
- `alignment.py` delegates to `matching.py` (retained until phase 4 cleanup)

**Tests:** `tests/test_matching.py` — matrix shape, Hungarian, threshold gating, mean score (toy B.1 style).

### Phase 4.2 — Extended primitive leaf compare ✅ *complete*

**Modules:** `matching.py` + `leaves.py`

- **`compare_primitive_list(pred, exp, …) -> LeafComparisonResult`** — public leaf API for paths like `tags`
- **`exact_primitive_similarity(pred, exp)`** — element-level helper in `leaves.py`
- Uses `matching.py`; aggregate **score** only (no layer S, no TP/FP/FN)
- Removed `align_primitive_lists` / `PrimitiveListAlignmentResult` from `alignment.py`

**Tests:** `tests/test_leaves.py` — `TestComparePrimitiveList` (toy example B, score only).

### Phase 4.3 — Object list row pairing ✅ *complete*

**Module:** `soda_mmqc/core/object_list_pairing.py`

- `align_object_rows`, `row_similarity`, `score_alignment_key`, `build_object_similarity_matrix`
- `ObjectListPairingResult`: `gold_to_pred`, `match_threshold`, `pair_similarities`, `pred_index_for_gold`
- `list_alignment` keys drive `s(i,j)`; each key uses **`fields`** profile for `{list_name}[].{key}`
- Hungarian via `matching.py`; pairing data only — **no** TP/FP/FN; **no** `by_list` here
- `alignment.py` deleted

**Tests:** `tests/test_object_list_pairing.py` — toy D row-pairing sections.

### Phase 4.4 — Layer S structural reporting ✅ *complete*

**Module:** `soda_mmqc/core/structural_reporting.py`

- `StructuralOutcome`: `correct_row`, `missing_row`, `spurious_row`
- `build_by_list(pairing, n_gold=…, n_pred=…) -> ByListResult` with `row_counts` + `rows` (`gold_index`, `pred_index`, `structural`, `similarity`)
- Below `τ`: **both** `missing_row` (gold) and `spurious_row` (pred)
- `list_recall`, `list_precision` from `row_counts`

**Tests:** `tests/test_structural_reporting.py` — toy A panels, D.2, D.4, D.5, below-threshold case.

---

## Phase 5 — Flat evaluator orchestration ✅ *complete*

**Module:** `soda_mmqc/core/evaluation.py` (reintroduced)

**Also:** `soda_mmqc/core/schema_discovery.py` — schema walk → leaf properties + object lists.

Pipeline from [evaluation-scoring.md](evaluation-scoring.md):

1. Flatten schema → leaf paths
2. For each **predictive** list-of-objects property (`list_alignment` present): `align_object_rows` → `build_by_list` → accumulate **`by_list`**. Structural lists: positional join only (no layer S).
3. Score extended primitives via **`leaves.compare_primitive_list`** (one instance per array leaf)
4. Score row leaves at gold indices (`leaves.py` scalar compares)
5. Apply layers 1 and 2 (`applicability_and_matching.py`)
6. Emit `instances` + `by_list` + `by_property`

**Tests:** `tests/test_schema_discovery.py`, `tests/test_evaluation.py` — toy examples A–D; golden `by_property` and `by_list` snapshots in `tests/fixtures/evaluation_snapshots/`.

---

## Phase 5.1 — Structural vs predictive lists ✅ *complete*

**Wiki:** [evaluation-scoring.md](evaluation-scoring.md#structural-vs-predictive-lists-schema-driven)

**Module:** `soda_mmqc/core/collation.py`

- `discover_collation_layout` — compare eval JSON shape to model `schema.json`; embedding prefix + structural vs predictive eval lists
- `build_eval_leaf_specs` — schema leaves remapped to eval paths; manifest-only structural row fields
- `validate_manifest_list_alignment` — `list_alignment` keys match resolved predictive paths; alignment keys ⊆ schema row properties
- `FlatEvaluator` — positional join for structural lists; Hungarian + `by_list` for schema predictive lists only
- `eval_manifest.alignment_keys_for` accepts `figures[].panels` or `figures.panels`

**Tests:** `tests/test_collation.py`, `tests/test_evaluation_collated.py`; toy tests unchanged (root embedding).

---

## Phase 5.2 — Layer 2 `mean_score` denominator ✅ *complete*

**Normative change** ([evaluation-scoring.md](evaluation-scoring.md#layer-2--matching)): `mean_score` is a **Layer 2 continuous rollup**. Average instance `score`s over **`correct_applicable` instances only** — same eligibility as `layer2_counts`. Exclude `correct_NA`, `withheld_applicable`, and `spurious_applicable` so N/A fields (matching `""`) do not dominate. If no eligible instances on a profiled property, `mean_score` = `0.0`. Unprofiled properties unchanged (mean over all instances).

**Delivered:**

| Location | Change |
|----------|--------|
| [`property_rollup.py`](../soda_mmqc/core/property_rollup.py) | `property_mean_score`, `instance_eligible_for_mean_score` — shared eligibility rule |
| [`evaluation.py`](../soda_mmqc/core/evaluation.py) `_summarize_by_property` | Uses `property_mean_score` with `profiled` flag |
| [`reporting/aggregate.py`](../soda_mmqc/reporting/aggregate.py) `aggregate_run` | Same filter when pooling `instances` across a run |
| [`reporting/plots.py`](../soda_mmqc/reporting/plots.py) | Scatter plots use `instance_eligible_for_mean_score` |
| [`tests/test_property_rollup.py`](../tests/test_property_rollup.py) | Unit tests for profiled vs unprofiled denominators |
| [`tests/test_evaluation.py`](../tests/test_evaluation.py) | Toy deltas: `item.status` N/A-only → `mean_score` `0.0`; D.2 `panels[].status` → `0.0` |
| [`soda_mmqc/docs/benchmarking.md`](../soda_mmqc/docs/benchmarking.md) | Documents applicable-only `mean_score` rule |

**Log:** [2026-06-18] impl entry in `thinking/log.md`.

---

## Phase 5.3 — Extended-primitive compare modes *(planned)*

**Normative change** ([evaluation-scoring.md](evaluation-scoring.md#list-of-primitives)): manifest `primitive_list_compare` selects how to score array-of-primitives leaves:

| Mode | Implementation target |
|------|----------------------|
| `align` (default) | Existing `compare_primitive_list` + `matching.py` |
| `positional` | New `compare_primitive_list_positional` — per-index element similarity, mean over `max(len)` |
| `join_string` | New `compare_primitive_list_join` — `join_separator`, optional `sort_before_join`, then `compare_strings` |

**Code changes required:**

| Location | Change |
|----------|--------|
| [`eval_manifest.py`](../soda_mmqc/core/eval_manifest.py) | Parse `primitive_list_compare`, `join_separator`, `sort_before_join` on `FieldProfile` |
| [`leaves.py`](../soda_mmqc/core/leaves.py) | `positional` and `join_string` compare functions |
| [`evaluation.py`](../soda_mmqc/core/evaluation.py) `_evaluate_extended_primitive` | Branch on `primitive_list_compare` |
| [`tests/test_leaves.py`](../tests/test_leaves.py) | Toy B.4 / B.5 style cases |
| [`tests/test_eval_manifest.py`](../tests/test_eval_manifest.py) | Manifest parsing for new keys |
| Checklist `eval-manifest.json` files | Set `positional` / `join_string` where needed (e.g. `micrograph-symbols-defined`) |
| [`soda_mmqc/docs/benchmarking.md`](../soda_mmqc/docs/benchmarking.md) | Document three modes |

**Out of scope:** per-element layer 2 on extended primitives; modelling parallel arrays as list-of-objects.

---

## Phase 6 — Integration

- Rewire `soda_mmqc/scripts/run.py`
- Migrate or replace legacy `tests/test_evaluate.py`, `test_compare_*.py` (old hierarchical `JSONEvaluator` tests)
- Add `eval-manifest.json` beside checklist schemas where missing

---

## Legacy tests (deferred)

These import the deleted hierarchical `JSONEvaluator` and stay **broken until phase 5–6**:

- `tests/test_evaluate.py`
- `tests/test_compare_strings.py`
- `tests/test_fuzzy_matching.py`
- `tests/test_compare_lists.py`
- `tests/test_compare_objects.py`

**Run completed work (phases 1–4):** `pytest tests/test_matching.py tests/test_leaves.py tests/test_eval_manifest.py tests/test_applicability_and_matching.py tests/test_object_list_pairing.py tests/test_structural_reporting.py`
