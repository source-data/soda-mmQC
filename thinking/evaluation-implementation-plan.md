# Evaluation implementation plan (progressive)

Normative contract: [evaluation-scoring.md](evaluation-scoring.md). Worked examples: [evaluation-toy-examples.md](evaluation-toy-examples.md).

Build the flat leaf comparator in small steps. Each step lands with **unit tests** before the next layer depends on it.

---

## Phase 1 — Primitive leaf comparison ✅ *complete*

**Module:** `soda_mmqc/core/leaves.py`

Pure functions: two values (+ minimal schema hints) → `LeafComparisonResult` with **`score ∈ [0, 1]`** only. No manifest layers, no list alignment, no `match_threshold` (that belongs in the outer comparator).

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

**Out of scope for phase 1:** `JSONEvaluator`, `instances` / `by_property`, manifest loading.

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

- **Layer 1 (applicability):** `correct_NA`, `spurious_applicable`, `withheld_applicable`, `correct_applicable`
- **Layer 2 (matching):** `TP`/`FP`/`FN`/`TN`, `match`/`mismatch` per `matching_metric`

**Tests:** `tests/test_applicability_and_matching.py` — mirror [evaluation-toy-examples.md](evaluation-toy-examples.md) sections C (status/label) and graded_string threshold cases.

---

## Phase 4a — List of primitives alignment ✅ *complete*

**Module:** `soda_mmqc/core/alignment.py` — `align_primitive_lists`, `build_similarity_matrix`, `exact_primitive_similarity`

- Pairwise element similarity → Hungarian assignment
- Pairs below `match_threshold` → alignment FP/FN; aggregate **score** over `max(len(exp), len(pred))`

**Tests:** `tests/test_alignment_primitive_lists.py` — toy example B

---

## Phase 4b — List of objects alignment ✅ *complete*

**Module:** `soda_mmqc/core/alignment.py` (extend)

- `align_object_lists`, `row_similarity`, `score_alignment_key`, `build_object_similarity_matrix`
- `list_alignment` keys (e.g. `panels → ["label"]`) drive which row fields participate in `s(i,j)`
- Each key uses the **`fields`** profile for `{list_name}[].{key}` (`string_compare`, `match_threshold` for graded strings; exact for enum/polarity)
- Hungarian assignment; pairs below mean key threshold → FP/FN; `gold_to_pred` mapping for downstream leaf scoring

**Tests:** `tests/test_alignment_object_lists.py` — toy example D (row alignment sections)

---

## Phase 5 — Flat evaluator orchestration

**Module:** `soda_mmqc/core/evaluation.py` (reintroduced)

Pipeline from [evaluation-scoring.md](evaluation-scoring.md):

1. Flatten schema → leaf paths
2. Align lists
3. Score instances (`leaves.py`)
4. Apply applicability and matching labels (`applicability_and_matching.py`)
5. Emit `instances` + `by_property`

**Tests:** full toy examples A–D; golden `by_property` JSON snapshots.

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

Run phases 1–4: `pytest tests/test_leaves.py tests/test_eval_manifest.py tests/test_applicability_and_matching.py tests/test_alignment_primitive_lists.py tests/test_alignment_object_lists.py`.
