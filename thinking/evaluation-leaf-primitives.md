# Leaf comparison: terminal primitive values

**Wiki:** catalog `[index.md](index.md)`; conventions `[README.md](README.md)`.

This page is about **terminal values** in schema-guided JSON comparison: the JSON types that do **not** contain nested objects or arrays. We compare them when the comparator reaches a path whose schema says `string`, `number`, `integer`, `boolean`, or (if we add it) `null`. **Strings are the richest case** because non-enum strings can use several similarity modes via the manifest; other primitives are **exact** today.

Normative scoring contract: [evaluation-scoring.md](evaluation-scoring.md). Worked examples: [evaluation-toy-examples.md](evaluation-toy-examples.md).

## Vocabulary: primitive vs scalar

- **JSON primitives** (common wording): `**string`**, `**number`**, `**integer**`, `**boolean**`, `**null**`. They are the values that sit at **leaves** of the JSON value tree (no `{}` or `[]` inside them).
- **“Scalar”** is used loosely in data and ML contexts to mean **one non-container value**. It often includes strings; sometimes people reserve “scalar” for numeric types only. In this wiki we say **primitive** or **leaf value** when we want to be precise.

A **leaf property** is a schema path to a primitive value (or array of primitives). Comparison produces **`score`** at these paths; manifest profiles add **layer 1** / **layer 2** labels per [evaluation-scoring.md](evaluation-scoring.md).

## Schema vs manifest at leaves

**Schema** (`schema.json`) constrains what the model may emit:

- `**type**` — which JSON family applies.
- `**enum**` (for strings) — closed vocabulary; pred outside `enum` → `score = 0`.
- Future: `**format**`, numeric bounds (`minimum` / `maximum`), `const`, etc.

**Manifest** (`eval-manifest.json`) constrains how we **score and report** profiled leaves:

- `matching_metric`, `na_values`, `positive_value` / `negative_value` — layers 1–2.
- `string_compare`, `match_threshold` — **only** on non-enum `graded_string` fields ([evaluation-scoring.md](evaluation-scoring.md#string_compare--how-the-leaf-score-is-computed)).

There is no global string metric: each free-text `graded_string` field declares its own `string_compare`. Schema `enum` fields are always exact; omit `string_compare` on those profiles.

## Vocabulary at leaves (only what leaves own)

| Term | Meaning at a **primitive leaf** |
|------|----------------------------------|
| `**score**` | Numeric quality in `**[0, 1]**` (exact → `0`/`1`; fuzzy / semantic → graded). Aggregated per property as `mean_score` in `by_property`. |
| **Validity / kind signals** | Hard failures (`enum` violation, type mismatch) → `**score == 0**` before layer reporting. |

**Normative split:** the leaf comparator returns **`score`** (and compared values). **Layer 1 / layer 2** (applicability, TP/FP/FN/TN, match/mismatch) come from the manifest profile on that leaf property — see [evaluation-scoring.md](evaluation-scoring.md). No container roll-up in the flat model.

## Strings

### Non-enum strings (`graded_string`)

For free `string` leaves with `matching_metric: graded_string`, declare **`string_compare`** in `fields` (`exact`, `fuzzy`, `semantic`). The leaf returns **`score`**; **`match_threshold`** turns that into layer 2 **match** / **mismatch** (`score >= τ`).

| `string_compare` | Behaviour |
|------------------|-----------|
| **exact** | Character-level equality → `0.0` or `1.0` |
| **fuzzy** | rapidfuzz `fuzz.ratio` (edit-distance) in `[0, 1]` |
| **semantic** | SentenceTransformer cosine similarity in `[0, 1]` |

`string_compare` is **not** a `defaults` key — set it on each free-text `graded_string` field that needs it.

**List alignment:** when a row field is an alignment key (e.g. `list_alignment.panels: ["label"]`), use that field’s `fields` profile (`string_compare`, `match_threshold`) to score `s(i,j)` for Hungarian matching. See [evaluation-scoring.md](evaluation-scoring.md#list-of-objects).

### Schema `enum` strings

If the schema lists `enum`, compare **exactly**; **do not** set `string_compare` on the manifest profile. Pred outside `enum` → **`score = 0`**.

- **N/A sentinel enums** (e.g. `["", "yes", "no"]` where `""` = not applicable): `""` is a **class**, not “empty wrong text.” Which literal is N/A and which is positive/negative is in **`eval-manifest.json`** (`na_values`, `positive_value`, `negative_value`). Layer 1 / 2: [evaluation-scoring.md](evaluation-scoring.md).
- **Multiclass enums** (`matching_metric: multiclass`): exact literal match → layer 2 **match** / **mismatch**; `mean_score` over instances.
- **Binary polarity enums** (`matching_metric: binary_polarity`): exact class match; layer 2 **TP** / **FP** / **FN** / **TN** on gold-applicable instances only. No `match_threshold`.

## Numbers and integers

**Strict equality** between JSON-parsed values (`pred_value == exp_value`) → **`score`** `1.0` or `0.0`. No manifest profile required unless you add layers later.

**Roadmap (design only):** tolerance (epsilon), `multipleOf`, integer-vs-float rules, or schema `const` — keyed off the schema fragment.

## Booleans

Same as numbers: **exact** equality, binary score.

## Null, missing keys, and empty strings

Follow **JSON** and **JSON Schema** literally — see [evaluation-scoring.md](evaluation-scoring.md#missing-null-and-empty-string):

- **Absent key** — `score = 0`.
- **`null`** on a string-only schema — `score = 0`.
- **`""`** — a string value; compare like any other string. For enums, `""` may be N/A via `na_values` (layer 1), not “missing”.

Do not use truthiness that conflates `""`, `null`, and absent keys.

If the schema allows `**null**` at a leaf (`type` union includes `"null"`), `null` vs `null` → full **`score`**; `null` vs non-null → `score = 0`.

## `score` vs layer 2 “match”

Leaves often return **partial credit** on `graded_string` paths (e.g. semantic similarity `0.4`). Layer 2 **match** / **mismatch** is **`score >= match_threshold`** — separate from `mean_score`, which averages raw scores across instances of that property. A high `score` below τ still counts as layer 2 **mismatch** but raises `mean_score`.

For `binary_polarity` and `multiclass` enums, layer 2 is **class identity**, not thresholded graded match.

## Output shape at a leaf (conceptual)

Every primitive leaf instance returns at minimum **`score` in `[0, 1]`**:

| Leaf type | How `score` is formed |
|-----------|------------------------|
| **free `string`** (`graded_string`) | `string_compare` from manifest → `[0, 1]` |
| **schema `enum` string** | Exact literal match → `0.0` or `1.0` |
| **number / integer** | Exact equality → `0.0` or `1.0` |
| **boolean** | Exact equality → `0.0` or `1.0` |

Optional leaf-local diagnostics (e.g. which comparator ran) are fine; they do not replace **`score`**.

The flat comparator stores one result per leaf **instance** path (`panels[1].status`, …) plus **`by_property`** summaries — no nested `field_scores` / `element_scores` trees ([evaluation-scoring.md](evaluation-scoring.md#comparator-output-required)).

**Separation of concerns:**

- **Leaf compare** — two values + schema fragment + optional manifest profile → **`score`**.
- **Comparator** — flatten paths, align lists (`list_alignment`), apply manifest layers, emit `instances` + `by_property`.

### `LeafComparisonResult` (target type)

**Not implemented yet** — target for the flat reimplementation. A small type returned by primitive compare functions:

- **`score`** in `[0, 1]`
- optional **`exp_value` / `pred_value`** (or the caller attaches these on the instance record)
- optional hard-failure flags (enum violation, type mismatch)

The outer comparator maps `LeafComparisonResult` + manifest profile → per-instance record (`path`, `layer1`, `layer2`, …). No hierarchical `field_scores` attachment at leaves.

**Direction:** primitive dispatch in `leaves.py`; applicability and matching labels in `applicability_and_matching.py`; list alignment in the evaluator. See [evaluation-implementation-plan.md](evaluation-implementation-plan.md).

## See also

- [Flat leaf scoring](evaluation-scoring.md) — manifest, layers, `list_alignment`, `by_property`.
- [Toy examples](evaluation-toy-examples.md) — worked gold/pred cases.
- [JSON evaluation vs open source](evaluation-json-vs-open-source.md) — why this leaf machinery stays project-specific.
